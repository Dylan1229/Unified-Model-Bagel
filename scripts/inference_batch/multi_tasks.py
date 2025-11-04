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


def run_text_to_image(
    inferencer: InterleaveInferencer,
    task: TaskSpec,
    default_shape: Tuple[int, int],
    enable_taylorseer: bool,
) -> Dict[str, Any]:
    params = dict(task.params or {})
    shape = tuple(params.pop("image_shape", params.pop("image_shapes", default_shape)))
    if len(shape) != 2:
        raise ValueError(f"image_shape must be [H, W], received: {shape}")

    cfg_interval = params.pop("cfg_interval", [0.4, 1.0])
    if isinstance(cfg_interval, (int, float)):
        cfg_interval = [float(cfg_interval), 1.0]

    inference_kwargs = dict(
        max_think_token_n=params.pop("max_think_token_n", 512),
        do_sample=params.pop("do_sample", False),
        text_temperature=params.pop("text_temperature", 0.3),
        cfg_text_scale=params.pop("cfg_text_scale", 4.0),
        cfg_img_scale=params.pop("cfg_img_scale", 1.5),
        cfg_interval=cfg_interval,
        timestep_shift=params.pop("timestep_shift", 3.0),
        num_timesteps=params.pop("num_timesteps", 50),
        cfg_renorm_min=params.pop("cfg_renorm_min", 0.0),
        cfg_renorm_type=params.pop("cfg_renorm_type", "global"),
        image_shapes=shape,
        enable_taylorseer=params.pop("enable_taylorseer", enable_taylorseer),
    )
    if params:
        raise ValueError(f"Unsupported parameters for text-to-image task: {params}")

    result = inferencer(text=task.prompt or "", think=task.think, **inference_kwargs)
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
    diffusion_executor: Optional[ThreadPoolExecutor] = None
    diffusion_stream: Optional[torch.cuda.Stream] = None
    pending_future: Optional[Future] = None
    pending_info: Optional[Dict[str, Any]] = None
    current_device = torch.cuda.current_device()

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