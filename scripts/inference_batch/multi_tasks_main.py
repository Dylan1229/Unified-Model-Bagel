import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import Future, ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
import torch.distributed as dist
from multi_task_function import (
    load_model,
    load_tasks,
    run_image_editing,
    run_image_understanding,
    run_text_to_image,
    setup_distributed,
    setup_seed,
    ensure_path,
    prepare_text_to_image_plan,
    execute_diffusion_async,
    )

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - pip requirement already present
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed BAGEL tasks (generation + understanding + editing).")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT", help="Directory that holds BAGEL weights/configs.")
    parser.add_argument("--tasks", type=Path, default=None, help="JSON or YAML file that describes the request queue.")
    parser.add_argument("--output", type=Path, default=Path("./results/multi_tasks"), help="Directory to store outputs.")
    parser.add_argument("--max_mem_per_gpu", default="80GiB", help="Maximum GPU memory per device for accelerate dispatch.")
    parser.add_argument("--default_shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(1024, 1024))
    parser.add_argument("--seed", type=int, default=42, help="Global fallback seed. Each task can override with its own `seed`.")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use for model sharding (default: 2).")
    parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer acceleration during generation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank provided by torchrun for distributed inference.")
    parser.add_argument("--distributed_backend", type=str, default="nccl", help="torch.distributed backend to use when launched with torchrun.")
    parser.add_argument(
        "--overlap_text_diffusion",
        action="store_true",
        help="Overlap text decoding of the next text-to-image task with the current diffusion phase using CUDA streams.",
    )
    return parser.parse_args()
    
def main() -> None:
    args = parse_args()
    dist_ctx = setup_distributed(args.local_rank)
    rank = dist_ctx.rank
    world_size = dist_ctx.world_size
    distributed = dist_ctx.enabled
    local_rank = dist_ctx.local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required but not available.")

    available_gpus = torch.cuda.device_count()
    if distributed:
        if local_rank < 0 or local_rank >= available_gpus:
            raise RuntimeError(
                f"Local rank {local_rank} exceeds available CUDA devices ({available_gpus})."
            )
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=args.distributed_backend, init_method="env://")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        if rank == 0:
            print(f"Distributed inference enabled with world size {world_size}.")
    else:
        torch.cuda.set_device(0)

    output_dir = args.output.expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    setup_seed(args.seed)
    tasks = load_tasks(args.tasks, output_dir)
    total_tasks = len(tasks)

    if distributed and args.num_gpus != 1:
        if rank == 0:
            print("Naive data parallel mode runs one GPU per process; ignoring --num_gpus.")

    device_ids = [torch.cuda.current_device()] if distributed else None
    offload_root = Path("/tmp") / f"offload_rank{rank}" if distributed else None

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        num_gpus=args.num_gpus,
        device_ids=device_ids,
        offload_dir=offload_root,
    )

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    default_shape = tuple(args.default_shape)

    assigned_indices = list(range(rank, total_tasks, world_size)) if distributed else list(range(total_tasks))
    assigned_total = len(assigned_indices)
    if distributed and assigned_total == 0:
        print(f"[rank {rank}] No tasks assigned to this rank.")

    summary_records: List[Tuple[int, Dict[str, Any]]] = []
    overlap_enabled = args.overlap_text_diffusion
    diffusion_executor: Optional[ThreadPoolExecutor] = None
    diffusion_stream: Optional[torch.cuda.Stream] = None
    pending_future: Optional[Future] = None
    pending_info: Optional[Dict[str, Any]] = None
    current_device = torch.cuda.current_device()

    if overlap_enabled:
        diffusion_executor = ThreadPoolExecutor(max_workers=1)
        diffusion_stream = torch.cuda.Stream(device=current_device)

    def finalize_pending() -> None:
        nonlocal pending_future, pending_info
        if pending_future is None or pending_info is None:
            return
        image = pending_future.result()
        image.save(pending_info["image_path"])
        summary_records.append(
            (
                pending_info["task_index"],
                {
                    "task_id": pending_info["task"].task_id,
                    "type": pending_info["task"].kind,
                    "prompt": pending_info["task"].prompt,
                    "image_path": str(pending_info["image_path"]),
                    "thinking_path": str(pending_info["thinking_path"]) if pending_info["thinking_path"] else None,
                },
            )
        )
        pending_future = None
        pending_info = None

    for local_idx, task_index in enumerate(assigned_indices, start=1):
        task = tasks[task_index]
        setup_seed(task.seed if task.seed is not None else args.seed)
        global_position = task_index + 1
        if distributed:
            print(
                f"[rank {rank}] [{local_idx}/{assigned_total}] (global {global_position}/{total_tasks}) "
                f"Running task '{task.task_id}' ({task.kind})"
            )
        else:
            print(f"[{global_position}/{total_tasks}] Running task '{task.task_id}' ({task.kind})")

        if task.kind in {"text2image", "text-to-image"}:
            if not task.prompt:
                raise ValueError(f"Task '{task.task_id}' requires a prompt.")

            if overlap_enabled:
                plan, thinking = prepare_text_to_image_plan(inferencer, task, default_shape, args.enable_taylorseer)
                image_path = ensure_path(task.output_image, task.task_id, output_dir, ".png")

                thinking_path: Optional[Path] = None
                if thinking:
                    thinking_path = ensure_path(
                        task.output_text,
                        f"{task.task_id}_thinking",
                        output_dir,
                        ".txt",
                    )
                    thinking_path.write_text(thinking, encoding="utf-8")

                finalize_pending()
                assert diffusion_executor is not None
                pending_future = diffusion_executor.submit(
                    execute_diffusion_async,
                    inferencer,
                    plan,
                    diffusion_stream,
                    current_device,
                )
                pending_info = {
                    "task_index": task_index,
                    "task": task,
                    "image_path": image_path,
                    "thinking_path": thinking_path,
                }
            else:
                result = run_text_to_image(inferencer, task, default_shape, args.enable_taylorseer)
                image = result.get("image")
                if image is None:
                    raise RuntimeError(f"No image returned for task '{task.task_id}'")
                image_path = ensure_path(task.output_image, task.task_id, output_dir, ".png")
                image.save(image_path)

                thinking = result.get("text")
                thinking_path: Optional[Path] = None
                if thinking:
                    thinking_path = ensure_path(
                        task.output_text,
                        f"{task.task_id}_thinking",
                        output_dir,
                        ".txt",
                    )
                    thinking_path.write_text(thinking, encoding="utf-8")

                summary_records.append(
                    (
                        task_index,
                        {
                            "task_id": task.task_id,
                            "type": task.kind,
                            "prompt": task.prompt,
                            "image_path": str(image_path),
                            "thinking_path": str(thinking_path) if thinking_path else None,
                        },
                    )
                )

        elif task.kind in {"image_understanding", "image-understanding", "vlm"}:
            if overlap_enabled:
                finalize_pending()
            result = run_image_understanding(inferencer, task)
            text_output = result.get("text")
            if not text_output:
                raise RuntimeError(f"No text returned for task '{task.task_id}'")
            text_path = ensure_path(task.output_text, task.task_id, output_dir, ".txt")
            text_path.write_text(text_output, encoding="utf-8")

            summary_records.append(
                (
                    task_index,
                    {
                        "task_id": task.task_id,
                        "type": task.kind,
                        "prompt": task.prompt,
                        "image_path": str(task.image_path) if task.image_path else None,
                        "text_path": str(text_path),
                    },
                )
            )

        elif task.kind in {"image_editing", "image-editing", "editing"}:
            if overlap_enabled:
                finalize_pending()
            result = run_image_editing(inferencer, task, args.enable_taylorseer)
            edited_image = result.get("image")
            if edited_image is None:
                raise RuntimeError(f"No edited image returned for task '{task.task_id}'")
            image_path = ensure_path(task.output_image, f"{task.task_id}_edited", output_dir, ".png")
            edited_image.save(image_path)

            thinking = result.get("text")
            thinking_path: Optional[Path] = None
            if thinking:
                thinking_path = ensure_path(
                    task.output_text,
                    f"{task.task_id}_thinking",
                    output_dir,
                    ".txt",
                )
                thinking_path.write_text(thinking, encoding="utf-8")

            summary_records.append(
                (
                    task_index,
                    {
                        "task_id": task.task_id,
                        "type": task.kind,
                        "prompt": task.prompt,
                        "source_image_path": str(task.image_path) if task.image_path else None,
                        "image_path": str(image_path),
                        "thinking_path": str(thinking_path) if thinking_path else None,
                    },
                )
            )

        else:
            raise ValueError(f"Unsupported task type '{task.kind}' in task '{task.task_id}'")

    if overlap_enabled:
        finalize_pending()
        if diffusion_executor is not None:
            diffusion_executor.shutdown(wait=True)

    if distributed:
        gathered: List[List[Tuple[int, Dict[str, Any]]]] = [None] * world_size
        dist.barrier()
        dist.all_gather_object(gathered, summary_records)
        if rank == 0:
            merged: List[Tuple[int, Dict[str, Any]]] = []
            for chunk in gathered:
                merged.extend(chunk)
            merged.sort(key=lambda item: item[0])
            summary = [entry for _, entry in merged]
            summary_path = output_dir / "summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")
        dist.barrier()
    else:
        summary = [entry for _, entry in summary_records]
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")

    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    main()