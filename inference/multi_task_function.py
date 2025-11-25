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
import torch.distributed as dist

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - pip requirement already present
    yaml = None



@dataclass
class TaskSpec:
    task_id: str
    kind: str
    prompt: Optional[str] = None
    image_path: Optional[Path] = None
    output_image: Optional[Path] = None
    output_text: Optional[Path] = None
    think: bool = False
    seed: Optional[int] = None
    params: Optional[Dict[str, Any]] = None
    image_shape: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

def sanitize_filename(text: str, max_length: int = 50) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text.strip())
    safe = safe.strip("._")
    if not safe:
        safe = "sample"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._")
    return safe


def ensure_path(path: Optional[Path], fallback_stem: str, base_dir: Path, suffix: str) -> Path:
    if path is None:
        name = sanitize_filename(fallback_stem)
        path = base_dir / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def parse_task_payload(
    payload: Dict[str, Any],
    output_dir: Path,
):
    task_id = payload.get("task_id") or payload.get("id") or ""
    prompt = payload.get("prompt")
    kind = (payload.get("type") or payload.get("kind") or "").lower()

    if not task_id:
        basis = prompt or kind or "task"
        task_id = sanitize_filename(basis)

    image_path: Optional[Path] = None
    if payload.get("image"):
        image_path = Path(payload["image"]).expanduser().resolve()

    output_image = payload.get("output_image") or payload.get("save_image_to")
    output_text = payload.get("output_text") or payload.get("save_text_to")

    output_image_path = Path(output_image).expanduser().resolve() if output_image else None
    output_text_path = Path(output_text).expanduser().resolve() if output_text else None

    raw_shape = payload.get("shape")
    image_shape: Optional[Tuple[int, int]] = None
    if raw_shape is not None:
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise ValueError(f"`shape` must be a pair of integers, received: {raw_shape}")
        image_shape = (int(raw_shape[0]), int(raw_shape[1]))

    return TaskSpec(
        task_id=task_id,
        kind=kind,
        prompt=prompt,
        image_path=image_path,
        output_image=output_image_path,
        output_text=output_text_path,
        think=bool(payload.get("think", False)),
        seed=payload.get("seed"),
        params=payload.get("params"),
        image_shape=image_shape,
    )




def build_text_to_image_kwargs(
    task: TaskSpec,
    default_shape: Tuple[int, int],
    enable_taylorseer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = dict(task.params or {})
    shape_source: Tuple[int, int] = task.image_shape or default_shape
    shape_value = params.pop("image_shape", params.pop("image_shapes", shape_source))
    shape = tuple(int(dim) for dim in shape_value)
    if len(shape) != 2:
        raise ValueError(f"image_shape must be [H, W], received: {shape}")

    cfg_interval = params.pop("cfg_interval", [0.4, 1.0])
    if isinstance(cfg_interval, (int, float)):
        cfg_interval = [float(cfg_interval), 1.0]
    elif isinstance(cfg_interval, tuple):
        cfg_interval = list(cfg_interval)

    inference_kwargs = dict(
        max_think_token_n=params.pop("max_think_token_n", 512),
        do_sample=params.pop("do_sample", False),
        text_temperature=params.pop("text_temperature", 0.3),
        cfg_text_scale=params.pop("cfg_text_scale", 4.0),
        cfg_img_scale=params.pop("cfg_img_scale", 1.5),
        cfg_interval=tuple(cfg_interval),
        timestep_shift=params.pop("timestep_shift", 3.0),
        num_timesteps=params.pop("num_timesteps", 50),
        cfg_renorm_min=params.pop("cfg_renorm_min", 0.0),
        cfg_renorm_type=params.pop("cfg_renorm_type", "global"),
        enable_taylorseer=params.pop("enable_taylorseer", enable_taylorseer),
    )
    if params:
        raise ValueError(f"Unsupported parameters for text-to-image task: {params}")

    call_kwargs = dict(inference_kwargs)
    call_kwargs["image_shapes"] = shape

    plan_kwargs = dict(inference_kwargs)
    plan_kwargs["image_shape"] = shape

    return call_kwargs, plan_kwargs

# ********************************** Build args for image editing and generation ****************************** #

def build_image_editing_kwargs(task: TaskSpec, enable_taylorseer: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if task.image_path is None:
        raise ValueError(f"Task '{task.task_id}' requires an `image` field.")
    if not task.image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {task.image_path}")

    params = dict(task.params or {})

    cfg_interval = params.pop("cfg_interval", [0.0, 1.0])
    if isinstance(cfg_interval, (int, float)):
        cfg_interval = [float(cfg_interval), 1.0]
    elif isinstance(cfg_interval, tuple):
        cfg_interval = list(cfg_interval)
    if not (isinstance(cfg_interval, list) and len(cfg_interval) == 2):
        raise ValueError(f"cfg_interval must be a pair of floats, received: {cfg_interval}")

    inference_kwargs = dict(
        max_think_token_n=params.pop("max_think_token_n", 1000),
        do_sample=params.pop("do_sample", True),
        text_temperature=params.pop("text_temperature", 1.0),
        cfg_text_scale=params.pop("cfg_text_scale", 4.0),
        cfg_img_scale=params.pop("cfg_img_scale", 2.0),
        cfg_interval=tuple(cfg_interval),
        timestep_shift=params.pop("timestep_shift", 3.0),
        num_timesteps=params.pop("num_timesteps", 50),
        cfg_renorm_min=params.pop("cfg_renorm_min", 0.0),
        cfg_renorm_type=params.pop("cfg_renorm_type", "text_channel"),
        enable_taylorseer=params.pop("enable_taylorseer", enable_taylorseer),
    )

    if params:
        raise ValueError(f"Unsupported parameters for image-editing task: {params}")

    plan_kwargs = dict(inference_kwargs)
    plan_kwargs["image_path"] = task.image_path

    return inference_kwargs, plan_kwargs

# ********************************** Task Runners ****************************** #

def run_text_to_image(
    inferencer: InterleaveInferencer,
    task: TaskSpec,
    default_shape: Tuple[int, int],
    enable_taylorseer: bool,
) -> Dict[str, Any]:
    call_kwargs, _ = build_text_to_image_kwargs(task, default_shape, enable_taylorseer)
    result = inferencer(text=task.prompt or "", think=task.think, **call_kwargs)
    return result

def run_image_editing(inferencer: InterleaveInferencer, task: TaskSpec, enable_taylorseer: bool):
    inference_kwargs, _ = build_image_editing_kwargs(task, enable_taylorseer)

    with Image.open(task.image_path) as image:
        image = image.convert("RGB")
        result = inferencer(image=image, text=task.prompt or "", think=task.think, **inference_kwargs)
    return result

def run_image_understanding(inferencer: InterleaveInferencer, task: TaskSpec):
    if task.image_path is None:
        raise ValueError(f"Task '{task.task_id}' requires an `image` field.")
    if not task.image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {task.image_path}")

    params = dict(task.params or {})
    inference_kwargs = dict(
        do_sample=params.pop("do_sample", False),
        text_temperature=params.pop("text_temperature", 0.3),
        max_think_token_n=params.pop("max_think_token_n", 512),
    )
    if params:
        raise ValueError(f"Unsupported parameters for image-understanding task: {params}")

    image = pil_img2rgb(Image.open(task.image_path).convert("RGB"))
    result = inferencer(image=image, text=task.prompt or "", think=task.think, understanding_output=True, **inference_kwargs)
    return result
