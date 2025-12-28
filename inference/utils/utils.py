import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

from data.data_utils import add_special_tokens
from data.transforms import ImageTransform
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
except ImportError:
    yaml = None


# ****************************************************************************
# * DATA STRUCTURES                               *
# ****************************************************************************

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


# ****************************************************************************
# * PATH & FILE HELPERS                             *
# ****************************************************************************

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


# ****************************************************************************
# * TASK PARSING                                 *
# ****************************************************************************

def parse_task_payload(payload: Dict[str, Any], output_dir: Path) -> TaskSpec:
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


def load_tasks(task_file: Optional[Path], output_dir: Path) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    
    # 1. Fallback to Demo tasks if no file provided
    if task_file is None:
        demo_tasks = [
            {
                "task_id": "demo-text2image",
                "type": "text2image",
                "prompt": "A futuristic tram gliding through a neon-lit city at dusk, cinematic lighting, wide shot.",
                "think": True,
                "params": {"max_think_token_n": 512, "cfg_text_scale": 4.0, "cfg_interval": 0.4},
            },
            {
                "task_id": "demo-image-understanding",
                "type": "image_understanding",
                "prompt": "Describe the scene and summarize why it is humorous.",
                "image": str((".." / "images" / "bike.jpg").resolve()),
                "think": False,
                "params": {"do_sample": False},
            },
        ]
        for entry in demo_tasks:
            tasks.append(parse_task_payload(entry, output_dir))
        return tasks

    # 2. Load from File
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


# ****************************************************************************
# * SETUP & INITIALIZATION                             *
# ****************************************************************************

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


# ****************************************************************************
# * MODEL LOADING                                 *
# ****************************************************************************

def _helper_device_to_str(value: Any) -> Optional[str]:
    """Helper to convert torch devices or ints to string representation."""
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
    
    # [Optim] A100/H100 optimization
    torch.set_float32_matmul_precision('high')

    # 1. Device Selection Logic
    if device_ids is not None:
        if len(device_ids) == 0:
            raise ValueError("device_ids must contain at least one GPU index.")
        normalized_ids = sorted(set(int(idx) for idx in device_ids))
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

    # 2. Load Configs
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    vae_model = vae_model.to(f"cuda:{device_ids[0]}", dtype=torch.float32).eval()

    bagel_config = BagelConfig(
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

    # 3. Initialize Empty Weights
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # 4. Infer Device Map
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in device_ids},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    # Force specific modules to share devices
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    
    num_selected_gpus = len(device_ids)
    first_device = _helper_device_to_str(device_map.get(same_device_modules[0])) or f"cuda:{device_ids[0]}"
    
    if num_selected_gpus == 1:
        # If single GPU, everything goes to first_device if not mapped
        for module in same_device_modules:
            device_map[module] = _helper_device_to_str(device_map.get(module)) or first_device
    else:
        # If multi-GPU, force modules to follow the first one found
        for module in same_device_modules:
            if module in device_map:
                device_map[module] = first_device

    # 5. Handle Offload Directory (PID Fix)
    if offload_dir is None:
        # Use a unique temporary directory for this specific process (PID)
        offload_path = Path(f"/tmp/offload_{os.getpid()}")
    else:
        offload_path = Path(offload_dir)
    offload_path.mkdir(parents=True, exist_ok=True)

    # 6. Load Checkpoint
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=str(offload_path),
    ).eval()

    # 7. Reporting
    used_devices = sorted({
        val if isinstance(val, str) and val.startswith("cuda") else f"cuda:{val}" 
        for val in device_map.values() if isinstance(val, (str, int))
    })
    print(f"Model shards placed on GPUs: {used_devices or [f'cuda:{i}' for i in device_ids]}")

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids