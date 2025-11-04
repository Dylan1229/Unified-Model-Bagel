import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - pip requirement already present
    yaml = None
from multi_task_function import (
    run_image_editing,
    run_image_understanding,
    run_text_to_image,
    setup_seed,
    load_tasks,
    ensure_path
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed BAGEL tasks (generation + understanding + editing).")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT", help="Directory that holds BAGEL weights/configs.")
    parser.add_argument("--tasks", type=Path, default=None, help="JSON or YAML file that describes the request queue.")
    parser.add_argument("--output", type=Path, default=Path("./results/multi_tasks"), help="Directory to store outputs.")
    parser.add_argument("--max_mem_per_gpu", default="80GiB", help="Maximum GPU memory per device for accelerate dispatch.")
    parser.add_argument("--default_shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(1024, 1024))
    parser.add_argument("--seed", type=int, default=42, help="Global fallback seed. Each task can override with its own `seed`.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for model sharding (default: 4).")
    parser.add_argument("--enable_taylorseer", action="store_true", help="Enable TaylorSeer acceleration during generation.")
    return parser.parse_args()





def main() -> None:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(args.seed)
    tasks = load_tasks(args.tasks, output_dir)

    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
        num_gpus=args.num_gpus,
    )

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    summary: List[Dict[str, Any]] = []
    default_shape = tuple(args.default_shape)

    for idx, task in enumerate(tasks, start=1):
        setup_seed(task.seed if task.seed is not None else args.seed)
        print(f"[{idx}/{len(tasks)}] Running task '{task.task_id}' ({task.kind})")

        if task.kind in {"text2image", "text-to-image"}:
            if not task.prompt:
                raise ValueError(f"Task '{task.task_id}' requires a prompt.")
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

            summary.append(
                {
                    "task_id": task.task_id,
                    "type": task.kind,
                    "prompt": task.prompt,
                    "image_path": str(image_path),
                    "thinking_path": str(thinking_path) if thinking_path else None,
                }
            )

        elif task.kind in {"image_understanding", "image-understanding", "vlm"}:
            result = run_image_understanding(inferencer, task)
            text_output = result.get("text")
            if not text_output:
                raise RuntimeError(f"No text returned for task '{task.task_id}'")
            text_path = ensure_path(task.output_text, task.task_id, output_dir, ".txt")
            text_path.write_text(text_output, encoding="utf-8")

            summary.append(
                {
                    "task_id": task.task_id,
                    "type": task.kind,
                    "prompt": task.prompt,
                    "image_path": str(task.image_path) if task.image_path else None,
                    "text_path": str(text_path),
                }
            )

        elif task.kind in {"image_editing", "image-editing", "editing"}:
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

            summary.append(
                {
                    "task_id": task.task_id,
                    "type": task.kind,
                    "prompt": task.prompt,
                    "source_image_path": str(task.image_path) if task.image_path else None,
                    "image_path": str(image_path),
                    "thinking_path": str(thinking_path) if thinking_path else None,
                }
            )

        else:
            raise ValueError(f"Unsupported task type '{task.kind}' in task '{task.task_id}'")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
