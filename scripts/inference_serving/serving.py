import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inferencer import InterleaveInferencer
from scripts.inference_batch.multi_tasks import (
    TaskSpec,
    ensure_path,
    load_model,
    load_tasks,
    run_image_editing,
    run_image_understanding,
    run_text_to_image,
    setup_seed,
)


def _parse_device_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    values: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values or None


@dataclass
class ArrivalEvent:
    arrival_time: float
    task_index: int
    task: TaskSpec


class PoissonArrivalSimulator:
    def __init__(self, lam: float, seed: Optional[int] = None) -> None:
        if lam <= 0.0:
            raise ValueError("poisson_lambda must be > 0.")
        self.lam = lam
        self.rng = np.random.default_rng(seed)

    def schedule(self, tasks: Iterable[TaskSpec]) -> List[ArrivalEvent]:
        task_list = list(tasks)
        if not task_list:
            return []

        intervals = self.rng.poisson(lam=self.lam, size=len(task_list)).astype(np.float64)
        arrival_times = intervals.cumsum()
        events: List[ArrivalEvent] = []
        for idx, (arrival_time, task) in enumerate(zip(arrival_times, task_list)):
            events.append(ArrivalEvent(float(arrival_time), idx, task))
        return events


class TaskServingEngine:
    def __init__(
        self,
        model_path: str,
        max_mem_per_gpu: str,
        num_gpus: int,
        device_ids: Optional[List[int]],
        offload_dir: Optional[Path],
        default_shape: Tuple[int, int],
        enable_taylorseer: bool,
        output_dir: Path,
        fallback_seed: Optional[int],
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device is required but not available.")

        if device_ids:
            torch.cuda.set_device(device_ids[0])
        else:
            torch.cuda.set_device(0)

        offload_path = offload_dir.resolve() if offload_dir else None
        if offload_path is not None:
            offload_path.mkdir(parents=True, exist_ok=True)

        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
            model_path=model_path,
            max_mem_per_gpu=max_mem_per_gpu,
            num_gpus=num_gpus,
            device_ids=device_ids,
            offload_dir=offload_path,
        )

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )
        self.default_shape = default_shape
        self.enable_taylorseer = enable_taylorseer
        self.output_dir = output_dir
        self.fallback_seed = fallback_seed

    def process_task(self, task: TaskSpec) -> Dict[str, Any]:
        seed = task.seed if task.seed is not None else self.fallback_seed
        setup_seed(seed)
        if task.kind in {"text2image", "text-to-image"}:
            if not task.prompt:
                raise ValueError(f"Task '{task.task_id}' requires a prompt.")
            result = run_text_to_image(self.inferencer, task, self.default_shape, self.enable_taylorseer)
            image = result.get("image")
            if image is None:
                raise RuntimeError(f"No image returned for task '{task.task_id}'.")
            image_path = ensure_path(task.output_image, task.task_id, self.output_dir, ".png")
            image.save(image_path)

            thinking_path: Optional[Path] = None
            thinking_text = result.get("text")
            if thinking_text:
                thinking_path = ensure_path(
                    task.output_text,
                    f"{task.task_id}_thinking",
                    self.output_dir,
                    ".txt",
                )
                thinking_path.write_text(thinking_text, encoding="utf-8")

            return {
                "task_id": task.task_id,
                "type": task.kind,
                "prompt": task.prompt,
                "image_path": str(image_path),
                "thinking_path": str(thinking_path) if thinking_path else None,
            }

        if task.kind in {"image_understanding", "image-understanding", "vlm"}:
            result = run_image_understanding(self.inferencer, task)
            text_output = result.get("text")
            if not text_output:
                raise RuntimeError(f"No text returned for task '{task.task_id}'.")
            text_path = ensure_path(task.output_text, task.task_id, self.output_dir, ".txt")
            text_path.write_text(text_output, encoding="utf-8")
            return {
                "task_id": task.task_id,
                "type": task.kind,
                "prompt": task.prompt,
                "image_path": str(task.image_path) if task.image_path else None,
                "text_path": str(text_path),
            }

        if task.kind in {"image_editing", "image-editing", "editing"}:
            result = run_image_editing(self.inferencer, task, self.enable_taylorseer)
            edited_image = result.get("image")
            if edited_image is None:
                raise RuntimeError(f"No edited image returned for task '{task.task_id}'.")
            image_path = ensure_path(task.output_image, f"{task.task_id}_edited", self.output_dir, ".png")
            edited_image.save(image_path)

            thinking_path: Optional[Path] = None
            thinking_text = result.get("text")
            if thinking_text:
                thinking_path = ensure_path(
                    task.output_text,
                    f"{task.task_id}_thinking",
                    self.output_dir,
                    ".txt",
                )
                thinking_path.write_text(thinking_text, encoding="utf-8")

            return {
                "task_id": task.task_id,
                "type": task.kind,
                "prompt": task.prompt,
                "source_image_path": str(task.image_path) if task.image_path else None,
                "image_path": str(image_path),
                "thinking_path": str(thinking_path) if thinking_path else None,
            }

        raise ValueError(f"Unsupported task type '{task.kind}' in task '{task.task_id}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve BAGEL multi-modal tasks with Poisson arrival simulation.")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT", help="Directory that holds BAGEL weights/configs.")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to a JSON file describing the request queue.")
    parser.add_argument("--output", type=Path, default=Path("./results/serving"), help="Directory to store outputs.")
    parser.add_argument("--max_mem_per_gpu", default="80GiB", help="Maximum GPU memory per device for accelerate dispatch.")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use for model sharding (default: 2).")
    parser.add_argument("--device_ids", type=str, default=None, help="Comma separated list of GPU indices to pin shards on.")
    parser.add_argument("--offload_dir", type=Path, default=None, help="Directory for accelerate offload buffers.")
    parser.add_argument("--default_shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(1024, 1024))
    parser.add_argument("--seed", type=int, default=42, help="Global seed controlling RNGs.")
    parser.add_argument("--poisson_lambda", type=float, default=1.0, help="Lambda parameter for the Poisson arrival process.")
    parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer acceleration during generation.")
    parser.add_argument("--wall_clock", action="store_true", help="Sleep between arrivals to emulate wall-clock latency.")
    parser.add_argument("--time_scale", type=float, default=1.0, help="Scale factor applied when --wall_clock is enabled.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle tasks before generating arrival times.")
    return parser.parse_args()


def main() -> None:
    test=parse_args()
    args = parse_args()
    if args.poisson_lambda <= 0.0:
        raise ValueError("poisson_lambda must be > 0.")

    tasks_path = args.tasks.expanduser().resolve()
    if tasks_path.suffix.lower() != ".json":
        raise ValueError(f"serving.py expects a JSON task file; received '{tasks_path.suffix}'.")

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(args.seed)

    tasks = load_tasks(tasks_path, output_dir)
    if args.shuffle and tasks:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(tasks)

    simulator = PoissonArrivalSimulator(args.poisson_lambda, seed=args.seed)
    schedule = simulator.schedule(tasks)

    if not schedule:
        summary_path = output_dir / "summary.json"
        summary_path.write_text("[]", encoding="utf-8")
        print("No tasks available; wrote empty summary.")
        return

    device_ids = _parse_device_ids(args.device_ids)

    offload_dir = args.offload_dir.expanduser().resolve() if args.offload_dir else None

    engine = TaskServingEngine(
        model_path=args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        num_gpus=args.num_gpus,
        device_ids=device_ids,
        offload_dir=offload_dir,
        default_shape=tuple(args.default_shape),
        enable_taylorseer=args.enable_taylorseer,
        output_dir=output_dir,
        fallback_seed=args.seed,
    )

    summary: List[Dict[str, Any]] = []
    current_time = 0.0
    wall_clock_origin = time.perf_counter()

    for event in schedule:
        wait = max(0.0, event.arrival_time - current_time)
        if args.wall_clock and wait > 0.0:
            time.sleep(wait * max(args.time_scale, 0.0))
        current_time = event.arrival_time
        simulated_now = current_time

        print(f"[t={simulated_now:.2f}] Dispatching task {event.task_index + 1}/{len(schedule)} ({event.task.task_id})")
        start = time.perf_counter()
        record = engine.process_task(event.task)
        duration = time.perf_counter() - start

        record.update(
            {
                "arrival_time": float(event.arrival_time),
                "processing_seconds": float(duration),
            }
        )
        if args.wall_clock:
            record["wall_clock_elapsed"] = float(time.perf_counter() - wall_clock_origin)
        summary.append(record)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
