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

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}


def setup_seed(seed: Optional[int]) -> None:
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_filename(text: str, max_length: int = 50) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text.strip())
    safe = safe.strip("._")
    if not safe:
        safe = "sample"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._")
    return safe

def parse_task_payload(
    payload: Dict[str, Any],
    output_dir: Path,
) -> TaskSpec:
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
    )


def load_tasks(task_file: Optional[Path], output_dir: Path) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    if task_file is None:
        demo_tasks = [
            {
                "task_id": "demo-text2image",
                "type": "text2image",
                "prompt": "A futuristic tram gliding through a neon-lit city at dusk, cinematic lighting, wide shot.",
                "think": True,
                "params": {
                    "max_think_token_n": 512,
                    "cfg_text_scale": 4.0,
                    "cfg_interval": 0.4,
                    "num_timesteps": 50,
                },
            },
            {
                "task_id": "demo-image-understanding",
                "type": "image_understanding",
                "prompt": "Describe the scene and summarize why it is humorous.",
                "image": str((".." / "images" / "bike.jpg").resolve()),
                "think": False,
                "params": {
                    "max_think_token_n": 512,
                    "do_sample": False,
                },
            },
        ]
        for entry in demo_tasks:
            tasks.append(parse_task_payload(entry, output_dir))
        return tasks

    task_file = task_file.expanduser().resolve()
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    with task_file.open("r", encoding="utf-8") as handle:
        if task_file.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to parse YAML task files.")
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)

    entries = data if isinstance(data, list) else data.get("tasks")
    if not entries:
        raise ValueError(f"No tasks defined in {task_file}")

    for entry in entries:
        tasks.append(parse_task_payload(entry, output_dir))
    return tasks


def ensure_path(path: Optional[Path], fallback_stem: str, base_dir: Path, suffix: str) -> Path:
    if path is None:
        name = sanitize_filename(fallback_stem)
        path = base_dir / f"{name}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_text_to_image_kwargs(
    task: TaskSpec,
    default_shape: Tuple[int, int],
    enable_taylorseer: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = dict(task.params or {})
    shape = tuple(params.pop("image_shape", params.pop("image_shapes", default_shape)))
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
        cfg_interval=cfg_interval,
        timestep_shift=params.pop("timestep_shift", 3.0),
        num_timesteps=params.pop("num_timesteps", 50),
        cfg_renorm_min=params.pop("cfg_renorm_min", 0.0),
        cfg_renorm_type=params.pop("cfg_renorm_type", "text_channel"),
        enable_taylorseer=params.pop("enable_taylorseer", enable_taylorseer),
    )

    if params:
        raise ValueError(f"Unsupported parameters for image-editing task: {params}")

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


def setup_distributed(local_rank_arg: int) -> DistributedContext:
    if not dist.is_available():
        return DistributedContext(False, 0, 1, max(local_rank_arg, 0))

    env_rank = os.environ.get("RANK")
    env_world = os.environ.get("WORLD_SIZE")
    if env_rank is None or env_world is None:
        return DistributedContext(False, 0, 1, max(local_rank_arg, 0))

    env_local = os.environ.get("LOCAL_RANK")
    local_rank = local_rank_arg if local_rank_arg >= 0 else int(env_local or env_rank)
    rank = int(env_rank)
    world_size = int(env_world)

    return DistributedContext(world_size > 1, rank, world_size, local_rank)


def setup_seed(seed: Optional[int]) -> None:
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_filename(text: str, max_length: int = 50) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text.strip())
    safe = safe.strip("._")
    if not safe:
        safe = "sample"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._")
    return safe


def load_model(
    model_path: str,
    max_mem_per_gpu: str = "80GiB",
    num_gpus: Optional[int] = None,
    device_ids: Optional[List[int]] = None,
    offload_dir: Optional[Path] = None,
):
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("CUDA device is required but not available.")

    if device_ids is not None and len(device_ids) == 0:
        raise ValueError("device_ids must contain at least one GPU index.")

    if device_ids is not None:
        normalized_ids: List[int] = sorted(set(int(idx) for idx in device_ids))
        for idx in normalized_ids:
            if idx < 0 or idx >= available_gpus:
                raise ValueError(f"Invalid GPU index {idx}. Available GPUs: {available_gpus}")
        device_ids = normalized_ids
    else:
        if num_gpus is None or num_gpus <= 0:
            num_gpus = available_gpus
        else:
            num_gpus = min(num_gpus, available_gpus)
        device_ids = list(range(num_gpus))

    num_selected_gpus = len(device_ids)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in device_ids},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    def _device_to_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, torch.device):
            if value.type == "cuda" and value.index is not None:
                return f"cuda:{value.index}"
            return value.type
        if isinstance(value, int):
            return f"cuda:{value}"
        return str(value)

    if num_selected_gpus == 1:
        default_device = f"cuda:{device_ids[0]}"
        first_device = _device_to_str(device_map.get(same_device_modules[0])) or default_device
        for module in same_device_modules:
            device_map[module] = _device_to_str(device_map.get(module)) or first_device
    else:
        first_device = _device_to_str(device_map.get(same_device_modules[0])) or f"cuda:{device_ids[0]}"
        for module in same_device_modules:
            if module in device_map:
                device_map[module] = first_device

    offload_path = Path(offload_dir or "/tmp/offload")
    offload_path.mkdir(parents=True, exist_ok=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=str(offload_path),
    ).eval()

    used_devices = set()
    for device in device_map.values():
        if isinstance(device, str):
            if device.startswith("cuda"):
                used_devices.add(device)
        elif isinstance(device, int):
            used_devices.add(f"cuda:{device}")
    used_devices = sorted(used_devices)
    print(f"Model shards placed on GPUs: {used_devices or [f'cuda:{i}' for i in device_ids]}")

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids
