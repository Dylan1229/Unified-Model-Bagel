import argparse
import sys
import time
from pathlib import Path
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inferencer import InterleaveInferencer
from inference.utils.multi_utils import (
    run_image_editing,
    run_text_to_image,
    run_image_understanding,
)
from inference.utils.utils import (
    ensure_path,
    load_model,
    load_tasks,
    setup_distributed,
    setup_seed,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed BAGEL tasks.")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT")
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("./results/multi_tasks"))
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument("--default_shape", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--enable_taylorseer", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--distributed_backend", type=str, default="nccl")
    parser.add_argument("--is_sliced", action="store_true", 
                        help="Enable patch-based diffusion")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size in pixels for patch-based diffusion")
    return parser.parse_args()

TEXT_TO_IMAGE_KINDS = {"text2image", "text-to-image", "txt2img", "txt-to-img"}
UNDERSTANDING_KINDS = {"image_understanding", "image-understanding", "vlm"}
IMAGE_EDITING_KINDS = {"image_editing", "image-editing", "editing"}

def prepare_output_dir(output_path: Path, rank: int, distributed: bool) -> Path:
    """Creates output directory."""
    
    output_path = output_path.expanduser().resolve()
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
    
    if distributed:
        dist.barrier() # Wait for rank 0 to create dir
    
    return output_path

def configure_devices(args, dist_ctx, available_gpus):
    """Configures devices for distributed training."""

    rank = dist_ctx.rank
    world_size = dist_ctx.world_size

    if dist_ctx.enabled:
        if dist_ctx.local_rank < 0 or dist_ctx.local_rank >= available_gpus:
            raise RuntimeError(f"Local rank {dist_ctx.local_rank} exceeds available GPUs.")
        torch.cuda.set_device(dist_ctx.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=args.distributed_backend, init_method="env://")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        device_ids = [torch.cuda.current_device()]
        if rank == 0:
            print(f"[Init] Distributed world size: {world_size}")
    else:
        requested = args.num_gpus if args.num_gpus > 0 else available_gpus
        num_use = min(requested, available_gpus)
        device_ids = list(range(num_use))
        torch.cuda.set_device(device_ids[0])
    
    return device_ids, rank, world_size

def main() -> None:
    args = parse_args()
    if args.is_sliced:
        print("Patch-based diffusion is enabled.")
    else:
        print("Patch-based diffusion is disabled.")
    
    # Set up distributed running
    dist_ctx = setup_distributed(args.local_rank)
    available_gpus = torch.cuda.device_count()
    device_ids, rank, world_size = configure_devices(args, dist_ctx, available_gpus)
    distributed = dist_ctx.enabled

    # Preparation
    output_dir = prepare_output_dir(args.output, rank, distributed)
    setup_seed(args.seed)

    # Load all tasks
    all_tasks = load_tasks(args.tasks, output_dir)
    
    # Assign tasks to each GPU
    # Each GPU takes a slice of the tasks: [0, 1, 2, 3] -> GPU0:[0, 2], GPU1:[1, 3]
    if distributed:
        my_tasks = all_tasks[rank::world_size]
        print(f"[Rank {rank}] Processing {len(my_tasks)}/{len(all_tasks)} tasks.")
    else:
        my_tasks = all_tasks
        print(f"[Single] Processing {len(my_tasks)} tasks.")

    # Unique offload dir for this process/rank
    offload_root = Path(f"/tmp/offload_rank{rank}_{dist_ctx.local_rank}") if distributed else None

    # Load Model
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        num_gpus=None,
        device_ids=device_ids,
        offload_dir=offload_root,
    )
    
    # Initialize Inferencer
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    
    # Main Execution Loop
    start_time = time.perf_counter()
    
    for i, task in enumerate(my_tasks):
        print(f"[Rank {rank}] Running Task {i+1}/{len(my_tasks)}: {task.task_id} ({task.kind})")
        
        try:
            # Image Generation
            if task.kind in TEXT_TO_IMAGE_KINDS:
                image, plan_text = run_text_to_image(
                    inferencer, 
                    task, 
                    tuple(args.default_shape), 
                    args.enable_taylorseer,
                    is_sliced=args.is_sliced,
                    patch_size=args.patch_size,
                )
                
                # Save Outputs
                if image:
                    path = ensure_path(task.output_image, task.task_id, output_dir, ".png")
                    image.save(path)
                if plan_text:
                    path = ensure_path(task.output_text, f"{task.task_id}_plan", output_dir, ".txt")
                    path.write_text(plan_text, encoding="utf-8")

            # Image Understanding
            elif task.kind in UNDERSTANDING_KINDS:
                generated_text = run_image_understanding(inferencer, task)
                
                if generated_text:
                    path = ensure_path(task.output_text, task.task_id, output_dir, ".txt")
                    path.write_text(generated_text, encoding="utf-8")
                    print(f"   > Output: {generated_text[:50]}...")

            # Image Editing
            elif task.kind in IMAGE_EDITING_KINDS:
                image, _ = run_image_editing(
                    inferencer, task, tuple(args.default_shape), args.enable_taylorseer,
                    is_sliced=args.is_sliced,
                    patch_size=args.patch_size,
                )
                if image:
                    path = ensure_path(task.output_image, task.task_id, output_dir, ".png")
                    image.save(path)
            
            # Unknown Task
            else:
                print(f"[Warning] Unknown task kind: {task.kind}")

        except Exception as e:
            print(f"[Error] Rank {rank} failed on task {task.task_id}: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.perf_counter() - start_time
    print(f"[Rank {rank}] Finished. Total time: {total_time:.2f}s")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()