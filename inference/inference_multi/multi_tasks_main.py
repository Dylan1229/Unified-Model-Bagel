import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from data.data_utils import add_special_tokens, pil_img2rgb
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
    build_image_editing_kwargs,
    build_text_to_image_kwargs,
    run_image_understanding
)

from inference.utils.utils import(
    setup_seed,
    setup_distributed,
    load_model,
    load_tasks,
    finalize_text_results,
    ensure_path
)

from scripts.inference_multi.scheduling import (
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
        "--parallel_mode",
        type=str,
        default=ParallelMode.DP.value,
        choices=[mode.value for mode in ParallelMode],
        help="Parallel scheduling strategy: DP,MP. While MP is not yet supported.",
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
    parallel_mode = ParallelMode(args.parallel_mode)
    if parallel_mode == ParallelMode.MP:
        raise NotImplementedError("Model parallel inference is not implemented yet.")

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
        device_ids = [diffusion_device]
    else:
        requested_gpus = args.num_gpus if args.num_gpus and args.num_gpus > 0 else available_gpus
        num_model_gpus = min(requested_gpus, available_gpus)
        if num_model_gpus <= 0:
            raise RuntimeError("Unable to allocate any CUDA device for inference.")
        device_ids = list(range(num_model_gpus))
        diffusion_device = device_ids[0]
        torch.cuda.set_device(diffusion_device)

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
    primary_text_to_image = primary_inferencer.text_to_image

    scheduler = build_scheduler(
        parallel_mode,
        inferencer_factory=get_inferencer,
        diffusion_device=diffusion_device,
    )

    default_shape = tuple(args.default_shape)

    assigned_indices = list(range(rank, total_tasks, world_size)) if distributed else list(range(total_tasks))
    assigned_total = len(assigned_indices)
    if distributed and assigned_total == 0:
        print(f"[rank {rank}] No tasks assigned to this rank.")

    
    summary_records = []
    generation_start = None
    generation_end = None

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

            finalize_text_results(scheduler.poll(), summary_records, output_dir)

            task_kind = (task.kind or "").lower()

            if task_kind in TEXT_TO_IMAGE_KINDS:
                if not task.prompt:
                    raise ValueError(f"Task '{task.task_id}' requires a prompt.")

                _, plan_kwargs = build_text_to_image_kwargs(task, default_shape, args.enable_taylorseer)
                plan_kwargs = dict(plan_kwargs)
                prompt_text = task.prompt or ""
                think_flag = bool(task.think)

                generation_start = time.perf_counter() if generation_start is None else generation_start
                request = TextToImageRequest(
                    task_index=task_index,
                    task=task,
                    prompt=prompt_text,
                    think=think_flag,
                    plan_kwargs=plan_kwargs,
                )
                scheduler.submit_text_to_image(request)

            elif task_kind in {"image_understanding", "image-understanding", "vlm"}:
                finalize_text_results(scheduler.poll(), summary_records, output_dir)
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

                if source_image is None:
                    raise ValueError(f"Task '{task.task_id}' requires an `image` field.")

                generation_start = time.perf_counter() if generation_start is None else generation_start
                with Image.open(source_image) as image_handle:
                    image_handle = image_handle.convert("RGB")
                    plan, thinking_text = primary_text_to_image.prepare_image_editing(
                        image=image_handle,
                        prompt=prompt_text,
                        think=think_flag,
                        device_id=diffusion_device,
                        **plan_kwargs,
                    )

                edited_image = primary_text_to_image.render_text_to_image_plan(plan, device_id=diffusion_device)
                finalize_text_results(
                    [
                        TextToImageResult(
                            task_index=task_index,
                            task=task,
                            image=edited_image,
                            thinking_text=thinking_text,
                        )
                    ],
                    summary_records,
                    output_dir,
                )

            else:
                raise ValueError(f"Unsupported task type '{task.kind}' in task '{task.task_id}'")

        finalize_text_results(scheduler.poll(), summary_records, output_dir)
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
