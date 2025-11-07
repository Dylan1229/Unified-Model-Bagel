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


def build_image_editing_kwargs(task: TaskSpec, enable_taylorseer: bool) -> Dict[str, Any]:
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
    return inference_kwargs


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
    inference_kwargs = build_image_editing_kwargs(task, enable_taylorseer)

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



    
# Our goal is to implement a class for diffusion stage pipeline with batching. We want  to run the batch stage of multiple requests together in a batch.
# The batch size will be decided by the argument, or by the previous profile result of images of different resolutions(we just implement by argument for now).
class ESyMReDSDXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline):
        self.pipeline = pipeline

    @staticmethod
    def from_pretrained(**kwargs):
        pretrained_model_name_or_path = kwargs.pop(
            "pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        torch_dtype = kwargs.pop("torch_dtype", torch.float16)
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, subfolder="unet"
        )
        
        unet = PatchUNet(unet)
        # print(unet)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch_dtype, unet=unet, **kwargs
        )
        return ESyMReDSDXLPipeline(pipeline)
    
    def get_profile(self, profile_dir):
        for name, module in self.pipeline.unet.named_modules():
            for subname, submodule in module.named_children():
                # if isinstance(submodule, SplitModule):
                if hasattr(submodule, "get_profile"):
                    submodule.get_profile(profile_dir)

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        is_sliced: bool = False,
        patch_size: int = 512,
        index: int = 0,
        input_indices: dict = None,
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.default_sample_size * self.pipeline.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.pipeline._guidance_scale = guidance_scale
        self.pipeline._guidance_rescale = guidance_rescale
        self.pipeline._clip_skip = clip_skip
        self.pipeline._cross_attention_kwargs = cross_attention_kwargs
        self.pipeline._denoising_end = denoising_end
        self.pipeline._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.pipeline.cross_attention_kwargs.get("scale", None) if self.pipeline.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.pipeline.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.pipeline.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.pipeline.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.unet.config.in_channels

        for key in latents:
            latents[key] = latents[key] * self.pipeline.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim

        add_time_ids = self.pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        base_offset = 0
        embeds = list()
        add_text_embeds_list = list()
        add_time_ids_list = list()
        if self.pipeline.do_classifier_free_guidance:
            for resolution in latents:
                embeds.append(torch.cat([negative_prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]], prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                add_text_embeds_list.append(torch.cat([negative_pooled_prompt_embeds[base_offset:base_offset+latents[resolution].shape[0]], add_text_embeds[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                add_time_ids_list.append(torch.cat([negative_add_time_ids[base_offset:base_offset+latents[resolution].shape[0]], add_time_ids[base_offset:base_offset+latents[resolution].shape[0]]], dim=0))
                base_offset = base_offset+latents[resolution].shape[0]
            prompt_embeds = torch.cat(embeds, dim=0)
            add_text_embeds = torch.cat(add_text_embeds_list, dim=0)
            add_time_ids = torch.cat(add_time_ids_list, dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.pipeline.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pipeline.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.pipeline.denoising_end is not None
            and isinstance(self.pipeline.denoising_end, float)
            and self.pipeline.denoising_end > 0
            and self.pipeline.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.pipeline.scheduler.config.num_train_timesteps
                    - (self.pipeline.denoising_end * self.pipeline.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipeline.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipeline.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.pipeline.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.pipeline.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.pipeline._num_timesteps = len(timesteps)
        # torch.cuda.synchronize()
        start = time.time()
        for resolution in latents:
            input_indices[resolution] = input_indices[resolution] + [id+"-1" for id in input_indices[resolution]]

        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipeline.interrupt:
                    continue
                latent_model_inputs = dict()
                # expand the latents if we are doing classifier free guidance
                
                length = 0
                for resolution in latents:
                    latent_model_inputs[resolution] = torch.cat([latents[resolution]] * 2) if self.pipeline.do_classifier_free_guidance else latents[resolution]
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    length += latent_model_inputs[resolution].shape[0]
                    latent_model_inputs[resolution] = self.pipeline.scheduler.scale_model_input(latent_model_inputs[resolution], t)
                    
                    # latent_model_inputs[resolution] = schedulers[resolution].scale_model_input(latent_model_inputs[resolution], t)
                # print(len(input_indices["768"]))
                # predict the noise residual
                
                # print(t)
                # print(length)
                t = [t for _ in range(length)]
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.pipeline.unet(
                    latent_model_inputs,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipeline.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    is_sliced=is_sliced,
                    patch_size=patch_size,
                    save_index=index,
                    input_indices=input_indices,
                )[0]

                # perform guidance
                for resolution, res_split_noise in noise_pred.items():
                    if self.pipeline.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = res_split_noise.chunk(2)
                        res_split_noise = noise_pred_uncond + self.pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents[resolution] = self.pipeline.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs).prev_sample

                    if self.pipeline.do_classifier_free_guidance and self.pipeline.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        res_split_noise = rescale_noise_cfg(res_split_noise, noise_pred_text, guidance_rescale=self.pipeline.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents[resolution] = schedulers[resolution].step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
                    latents[resolution] = self.pipeline.scheduler.step(res_split_noise, t, latents[resolution], **extra_step_kwargs, return_dict=False)[0]
                    self.pipeline.scheduler._step_index -= 1
                self.pipeline.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.pipeline.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # torch.cuda.synchronize()
        end = time.time()
        total_unet_time = end - start
        start = time.time()
        # return total_unet_time, total_unet_time, 0
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.pipeline.vae.dtype == torch.float16 and self.pipeline.vae.config.force_upcast
            images = dict()
            for resolution in latents:
                if needs_upcasting:
                    self.pipeline.upcast_vae()
                    latents[resolution] = latents[resolution].to(next(iter(self.pipeline.vae.post_quant_conv.parameters())).dtype)

                # unscale/denormalize the latents
                # denormalize with the mean and std if available and not None
                has_latents_mean = hasattr(self.pipeline.vae.config, "latents_mean") and self.pipeline.vae.config.latents_mean is not None
                has_latents_std = hasattr(self.pipeline.vae.config, "latents_std") and self.pipeline.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.pipeline.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.pipeline.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                    )
                    latents[resolution] = latents[resolution] * latents_std / self.pipeline.vae.config.scaling_factor + latents_mean
                else:
                    latents[resolution] = latents[resolution] / self.pipeline.vae.config.scaling_factor

                image = self.pipeline.vae.decode(latents[resolution], return_dict=False)[0]
                images[resolution] = image

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.pipeline.vae.to(dtype=torch.float16)
            else:
                image = latents

        if not output_type == "latent":
            for resolution in images:
            # apply watermark if available
                if self.pipeline.watermark is not None:
                    images[resolution] = self.pipeline.watermark.apply_watermark(images[resolution])

                images[resolution] = self.pipeline.image_processor.postprocess(images[resolution], output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()
        # torch.cuda.synchronize()
        end = time.time()
        total_post_process_time = end - start
        if not return_dict:
            return (images, total_unet_time, total_post_process_time,)

        return images, total_unet_time, total_post_process_time
        
        
        
        image,unet_time,_ = pipe(
        prompts[index * bs * 3: (index + 1) * bs * 3],
        num_inference_steps=50,
        guidance_scale=4.5,
        input_indices=input_indices, 
        index=index, latents=latents, 
        patch_size=256, is_sliced=True
    )
