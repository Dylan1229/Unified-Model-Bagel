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

TEXT_TO_IMAGE_KINDS = {"text2image", "text-to-image"}
IMAGE_EDITING_KINDS = {"image_editing", "image-editing", "editing"}



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


def run_text_to_image(
    inferencer: InterleaveInferencer,
    task: TaskSpec,
    default_shape: Tuple[int, int],
    enable_taylorseer: bool,
) -> Dict[str, Any]:
    call_kwargs, _ = build_text_to_image_kwargs(task, default_shape, enable_taylorseer)
    result = inferencer(text=task.prompt or "", think=task.think, **call_kwargs)
    return result


def run_image_editing(inferencer: InterleaveInferencer, task: TaskSpec, enable_taylorseer: bool) -> Dict[str, Any]:
    inference_kwargs, _ = build_image_editing_kwargs(task, enable_taylorseer)

    with Image.open(task.image_path) as image:
        image = image.convert("RGB")
        result = inferencer(image=image, text=task.prompt or "", think=task.think, **inference_kwargs)
    return result


def run_image_understanding(inferencer: InterleaveInferencer, task: TaskSpec) -> Dict[str, Any]:
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

@dataclass
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
