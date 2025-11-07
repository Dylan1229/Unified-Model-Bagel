import argparse
import copy
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import time
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from data.data_utils import add_special_tokens, pil_img2rgb
from inferencer import InterleaveInferencer, TextToImagePlan
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
    build_image_editing_kwargs,
    build_text_to_image_kwargs,
    ensure_path,
    load_model,
    load_tasks,
    run_image_understanding,
    setup_distributed,
    setup_seed,
)
from scripts.inference_serving.scheduling import (
    ParallelMode,
    TextToImageRequest,
    TextToImageResult,
    build_scheduler,
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
        "--parallel_method",
        type=str,
        default=ParallelMode.DATA_PARALLEL.value,
        choices=[mode.value for mode in ParallelMode],
        help="Parallel scheduling strategy: data_parallel (sequential) or staged_diffusion (decode/diffusion split).",
    )
    parser.add_argument(
        "--diffusion_gpu",
        type=int,
        default=0,
        help="GPU index dedicated to diffusion/rendering when using staged diffusion scheduling.",
    )
    parser.add_argument(
        "--decode_gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices for text decoding workers in staged diffusion scheduling.",
    )
    parser.add_argument(
        "--overlap_text_diffusion",
        action="store_true",
        help="(Deprecated) Enable staged diffusion scheduling. Use --parallel_method=staged_diffusion instead.",
    )
    parser.add_argument(
        "--batch_size_1024",
        type=int,
        default=1,
        help="Diffusion batch size for 1024x1024 generations.",
    )
    parser.add_argument(
        "--batch_size_768",
        type=int,
        default=1,
        help="Diffusion batch size for 768x768 generations.",
    )
    parser.add_argument(
        "--batch_size_512",
        type=int,
        default=1,
        help="Diffusion batch size for 512x512 generations.",
    )
    return parser.parse_args()


TEXT_TO_IMAGE_KINDS = {"text2image", "text-to-image"}
IMAGE_EDITING_KINDS = {"image_editing", "image-editing", "editing"}

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
    requested_mode_value = args.parallel_method
    if args.overlap_text_diffusion and requested_mode_value == ParallelMode.DATA_PARALLEL.value:
        requested_mode_value = ParallelMode.STAGED_DIFFUSION.value
        if rank == 0:
            print("WARNING: --overlap_text_diffusion is deprecated; use --parallel_method=staged_diffusion instead.")
    try:
        parallel_mode = ParallelMode(requested_mode_value)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported parallel method '{requested_mode_value}'.") from exc

    if distributed and parallel_mode != ParallelMode.DATA_PARALLEL:
        if rank == 0:
            print("Staged diffusion scheduling is unavailable in distributed mode. Falling back to data_parallel.")
        parallel_mode = ParallelMode.DATA_PARALLEL

    def parse_device_list(value: Optional[str]) -> List[int]:
        if value is None:
            return []
        devices: List[int] = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            devices.append(int(chunk))
        return devices

    def build_batch_size_resolver() -> Callable[[TextToImagePlan], int]:
        def _clamp(value: int) -> int:
            return max(1, int(value))

        resolution_map = {
            1024: _clamp(args.batch_size_1024),
            768: _clamp(args.batch_size_768),
            512: _clamp(args.batch_size_512),
        }

        def resolver(plan: TextToImagePlan) -> int:
            shape = getattr(plan, "image_shape", None)
            if shape is not None:
                dims = [int(shape[0]), int(shape[1])]
                for dim in dims:
                    batch_size = resolution_map.get(dim)
                    if batch_size and batch_size > 0:
                        return batch_size
            return 1

        return resolver

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
        diffusion_device = torch.cuda.current_device()
        decode_gpus: List[int] = []
    else:
        diffusion_device = args.diffusion_gpu
        if diffusion_device < 0 or diffusion_device >= available_gpus:
            raise ValueError(
                f"Diffusion GPU index {diffusion_device} is invalid for {available_gpus} available device(s)."
            )
        torch.cuda.set_device(diffusion_device)
        decode_gpus = []
        if parallel_mode == ParallelMode.STAGED_DIFFUSION:
            decode_gpus = parse_device_list(args.decode_gpus)
            if not decode_gpus:
                decode_gpus = [idx for idx in range(available_gpus) if idx != diffusion_device]
            decode_gpus = sorted({idx for idx in decode_gpus if idx != diffusion_device})
            for idx in decode_gpus:
                if idx < 0 or idx >= available_gpus:
                    raise ValueError(
                        f"Decode GPU index {idx} is invalid for {available_gpus} available device(s)."
                    )
            if not decode_gpus:
                raise ValueError("Staged diffusion scheduling requires at least one decode GPU distinct from the diffusion GPU.")

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

    if distributed:
        model_device_ids = [torch.cuda.current_device()]
    elif parallel_mode == ParallelMode.STAGED_DIFFUSION:
        model_device_ids = sorted({diffusion_device, *decode_gpus})
    else:
        requested_gpus = args.num_gpus if args.num_gpus and args.num_gpus > 0 else available_gpus
        num_model_gpus = min(requested_gpus, available_gpus)
        model_device_ids = list(range(num_model_gpus))
    if diffusion_device not in model_device_ids:
        raise ValueError(
            f"Diffusion GPU {diffusion_device} is not included in the model device map {model_device_ids}."
        )

    device_ids = [torch.cuda.current_device()] if distributed else model_device_ids
    offload_root = Path("/tmp") / f"offload_rank{rank}" if distributed else None

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        num_gpus=None,
        device_ids=device_ids,
        offload_dir=offload_root,
    )
    vae_model = vae_model.eval()

    inferencer_cache: Dict[int, InterleaveInferencer] = {}
    vae_cache: Dict[int, torch.nn.Module] = {}

    def get_inferencer(device_id: int) -> InterleaveInferencer:
        instance = inferencer_cache.get(device_id)
        if instance is None:
            if device_id not in vae_cache:
                localized_vae = copy.deepcopy(vae_model).to(f"cuda:{device_id}").eval()
                vae_cache[device_id] = localized_vae
            instance = InterleaveInferencer(
                model=model,
                vae_model=vae_cache[device_id],
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )
            inferencer_cache[device_id] = instance
        return instance

    primary_inferencer = get_inferencer(diffusion_device)
    batch_size_resolver = build_batch_size_resolver()

    if parallel_mode == ParallelMode.STAGED_DIFFUSION:
        scheduler = build_scheduler(
            ParallelMode.STAGED_DIFFUSION,
            inferencer_factory=get_inferencer,
            diffusion_device=diffusion_device,
            decode_devices=decode_gpus,
            default_batch_size=1,
            batch_size_resolver=batch_size_resolver,
        )
    else:
        scheduler = build_scheduler(
            ParallelMode.DATA_PARALLEL,
            inferencer_factory=get_inferencer,
            diffusion_device=diffusion_device,
            default_batch_size=1,
            batch_size_resolver=batch_size_resolver,
        )

    default_shape = tuple(args.default_shape)

    assigned_indices = list(range(rank, total_tasks, world_size)) if distributed else list(range(total_tasks))
    assigned_total = len(assigned_indices)
    if distributed and assigned_total == 0:
        print(f"[rank {rank}] No tasks assigned to this rank.")

    summary_records: List[Tuple[int, Dict[str, Any]]] = []
    generation_start: Optional[float] = None
    generation_end: Optional[float] = None

    def mark_generation_start() -> None:
        nonlocal generation_start
        if generation_start is None:
            generation_start = time.perf_counter()
    def finalize_text_results(results: List[TextToImageResult]) -> None:
        for outcome in results:
            if outcome.error:
                raise RuntimeError(
                    f"Diffusion task '{outcome.task.task_id}' failed."
                ) from outcome.error
            image = outcome.image
            if image is None:
                raise RuntimeError(f"No image returned for task '{outcome.task.task_id}'")

            task_kind = (outcome.task.kind or "").lower()
            if task_kind in TEXT_TO_IMAGE_KINDS:
                image_path = ensure_path(outcome.task.output_image, outcome.task.task_id, output_dir, ".png")
            elif task_kind in IMAGE_EDITING_KINDS:
                image_path = ensure_path(
                    outcome.task.output_image,
                    f"{outcome.task.task_id}_edited",
                    output_dir,
                    ".png",
                )
            else:
                image_path = ensure_path(outcome.task.output_image, outcome.task.task_id, output_dir, ".png")
            image.save(image_path)

            thinking_path: Optional[Path] = None
            if outcome.thinking_text:
                thinking_path = ensure_path(
                    outcome.task.output_text,
                    f"{outcome.task.task_id}_thinking",
                    output_dir,
                    ".txt",
                )
                thinking_path.write_text(outcome.thinking_text, encoding="utf-8")

            summary_payload: Dict[str, Any] = {
                "task_id": outcome.task.task_id,
                "type": outcome.task.kind,
                "prompt": outcome.task.prompt,
                "image_path": str(image_path),
                "thinking_path": str(thinking_path) if thinking_path else None,
            }
            if task_kind in IMAGE_EDITING_KINDS:
                summary_payload["source_image_path"] = str(outcome.task.image_path) if outcome.task.image_path else None

            summary_records.append(
                (
                    outcome.task_index,
                    summary_payload,
                )
            )

    try:
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

            finalize_text_results(scheduler.poll())

            task_kind = (task.kind or "").lower()

            if task_kind in TEXT_TO_IMAGE_KINDS:
                if not task.prompt:
                    raise ValueError(f"Task '{task.task_id}' requires a prompt.")

                _, plan_kwargs = build_text_to_image_kwargs(task, default_shape, args.enable_taylorseer)
                plan_kwargs = dict(plan_kwargs)
                prompt_text = task.prompt or ""
                think_flag = bool(task.think)

                mark_generation_start()
                request = TextToImageRequest(
                    task_index=task_index,
                    task=task,
                    build_plan=lambda inferencer, device_id, prompt=prompt_text, think=think_flag, kwargs=plan_kwargs: inferencer.prepare_text_to_image(
                        prompt,
                        think=think,
                        device_id=device_id,
                        **kwargs,
                    ),
                )
                scheduler.submit_text_to_image(request)

            elif task_kind in {"image_understanding", "image-understanding", "vlm"}:
                finalize_text_results(scheduler.wait_all())
                result = run_image_understanding(primary_inferencer, task)
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

            elif task_kind in IMAGE_EDITING_KINDS:
                _, plan_kwargs = build_image_editing_kwargs(task, args.enable_taylorseer)
                plan_kwargs = dict(plan_kwargs)
                source_image = plan_kwargs.pop("image_path")
                prompt_text = task.prompt or ""
                think_flag = bool(task.think)

                def _build_edit_plan(
                    inferencer: InterleaveInferencer,
                    device_id: int,
                    *,
                    prompt: str,
                    think: bool,
                    kwargs: Dict[str, Any],
                    source_path: Path,
                ) -> Tuple[TextToImagePlan, Optional[str]]:
                    with Image.open(source_path) as image:
                        image = image.convert("RGB")
                        return inferencer.prepare_image_editing(
                            image=image,
                            prompt=prompt,
                            think=think,
                            device_id=device_id,
                            **kwargs,
                        )

                mark_generation_start()
                request = TextToImageRequest(
                    task_index=task_index,
                    task=task,
                    build_plan=lambda inferencer, device_id, prompt=prompt_text, think=think_flag, kwargs=plan_kwargs, src=source_image: _build_edit_plan(
                        inferencer,
                        device_id,
                        prompt=prompt,
                        think=think,
                        kwargs=kwargs,
                        source_path=src,
                    ),
                )
                scheduler.submit_text_to_image(request)

            else:
                raise ValueError(f"Unsupported task type '{task.kind}' in task '{task.task_id}'")

        finalize_text_results(scheduler.wait_all())
        if generation_start is not None and generation_end is None:
            generation_end = time.perf_counter()
    finally:
        scheduler.shutdown()

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
        summary_records.sort(key=lambda item: item[0])
        summary = [entry for _, entry in summary_records]
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")

    if generation_start is not None and generation_end is not None:
        elapsed = generation_end - generation_start
        message = f"Generation time (requests): {elapsed:.2f} seconds"
        if distributed:
            print(f"[rank {rank}] {message}")
        else:
            print(message)

    if dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    main()
