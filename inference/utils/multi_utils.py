import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.utils.utils import TaskSpec, ensure_path

# Default fallback constants
DEFAULT_CFG_INTERVAL = (0.4, 1.0)
GEN_THINK_SYSTEM_PROMPT = "You are a helpful assistant. First, think about the request steps."

# --- Helper: Parameter Extraction ---
def get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    if params is None:
        return default
    return params.get(key, default)

def normalize_interval(raw: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if raw is None:
        return fallback
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return fallback

# --- Helper: Image Loading ---
def load_input_image(image_path: Optional[Path]) -> Optional[Image.Image]:
    if image_path is None or not image_path.exists():
        return None
    return Image.open(image_path).convert("RGB")

# ********************************** Task Runners ****************************** #

def run_text_to_image(inferencer, task: TaskSpec, default_shape: Tuple[int, int], enable_taylorseer: bool = False):
    """
    Handles Text-to-Image generation, including optional 'Thinking' step.
    """
    # 1. Initialize Context
    gen_context = inferencer.init_gen_context()
    cfg_text_ctx = inferencer.init_gen_context()
    cfg_img_ctx = inferencer.init_gen_context()

    # 2. Handle "Thinking" (Plan Generation) if requested
    think_text = None
    if task.think:
        gen_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
        cfg_img_ctx = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_ctx)
        
        think_text = inferencer.gen_text(
            gen_context,
            do_sample=bool(get_param(task.params, "do_sample", True)),
            temperature=float(get_param(task.params, "text_temperature", 0.3)),
            max_length=int(get_param(task.params, "max_think_token_n", 512)),
        )
        
        gen_context = inferencer.update_context_text(think_text, gen_context)
        cfg_img_ctx = inferencer.update_context_text(think_text, cfg_img_ctx)

    # 3. Prepare Image Generation Context
    prompt = task.prompt or ""
    gen_context = inferencer.update_context_text(prompt, gen_context)
    cfg_text_ctx = inferencer.update_context_text(prompt, inferencer.init_gen_context()) 
    cfg_img_ctx = inferencer.update_context_text(prompt, cfg_img_ctx)

    # 4. Generate Image
    image_shape = task.image_shape or default_shape
    cfg_interval = normalize_interval(get_param(task.params, "cfg_interval", None), DEFAULT_CFG_INTERVAL)

    image = inferencer.gen_image(
        image_shape,
        gen_context,
        cfg_text_precontext=cfg_text_ctx,
        cfg_img_precontext=cfg_img_ctx,
        cfg_text_scale=float(get_param(task.params, "cfg_text_scale", 4.0)),
        cfg_img_scale=float(get_param(task.params, "cfg_img_scale", 1.5)),
        cfg_interval=cfg_interval,
        timestep_shift=float(get_param(task.params, "timestep_shift", 3.0)),
        num_timesteps=int(get_param(task.params, "num_timesteps", 50)),
        cfg_renorm_min=float(get_param(task.params, "cfg_renorm_min", 0.0)),
        cfg_renorm_type=get_param(task.params, "cfg_renorm_type", "global"),
        enable_taylorseer=enable_taylorseer,
    )

    return image, think_text


def run_image_understanding(inferencer, task: TaskSpec):
    """
    Handles Image-to-Text (VLM) tasks.
    """
    input_image = load_input_image(task.image_path)
    if input_image is None:
        raise ValueError(f"Image not found for understanding task: {task.image_path}")

    # Determine Model Dtype (Fixes Float vs BFloat16 error)
    # If the model is wrapped in Accelerate, getting dtype might be tricky, so we check the first parameter
    try:
        model_dtype = next(inferencer.model.parameters()).dtype
    except:
        model_dtype = torch.bfloat16

    # 1. Initialize Context with Image
    gen_context = inferencer.init_gen_context()
    
    # [FIX] Cast is likely handled inside update_context_image via transforms, 
    # but we force autocast context just in case, or ensure inferencer handles it.
    # The safest way is to ensure the inferencer treats image input correctly.
    # Assuming update_context_image eventually calls model.forward, we wrap in autocast.
    
    with torch.autocast(device_type="cuda", dtype=model_dtype):
        gen_context = inferencer.update_context_image(input_image, gen_context)
    
    # 2. Add Prompt
    prompt = task.prompt or "Describe this image."
    gen_context = inferencer.update_context_text(prompt, gen_context)

    # 3. Generate Text
    generated_text = inferencer.gen_text(
        gen_context,
        do_sample=bool(get_param(task.params, "do_sample", False)),
        temperature=float(get_param(task.params, "text_temperature", 0.1)),
        max_length=int(get_param(task.params, "max_new_tokens", 512)),
    )
    
    return generated_text


def run_image_editing(inferencer, task: TaskSpec, default_shape: Tuple[int, int], enable_taylorseer: bool = False):
    """
    Handles Image Editing tasks (Image+Text -> Image).
    """
    input_image = load_input_image(task.image_path)
    if input_image is None:
        raise ValueError(f"Image not found for editing task: {task.image_path}")
    
    # Determine Model Dtype
    try:
        model_dtype = next(inferencer.model.parameters()).dtype
    except:
        model_dtype = torch.bfloat16

    # Resize input image to target shape if necessary
    target_shape = task.image_shape or default_shape
    if input_image.size != target_shape[::-1]: # PIL is W,H
         input_image = input_image.resize(target_shape[::-1], Image.LANCZOS)

    # 1. Initialize Context
    gen_context = inferencer.init_gen_context()
    cfg_text_ctx = inferencer.init_gen_context()
    cfg_img_ctx = inferencer.init_gen_context()

    # 2. Update Contexts with Input Image [FIX: Autocast for dtype mismatch]
    with torch.autocast(device_type="cuda", dtype=model_dtype):
        gen_context = inferencer.update_context_image(input_image, gen_context)
        cfg_img_ctx = inferencer.update_context_image(input_image, cfg_img_ctx)
    
    # 3. Update Contexts with Editing Prompt
    prompt = task.prompt or ""
    gen_context = inferencer.update_context_text(prompt, gen_context)
    cfg_text_ctx = inferencer.update_context_text(prompt, cfg_text_ctx)
    cfg_img_ctx = inferencer.update_context_text(prompt, cfg_img_ctx)

    # 4. Generate Edited Image
    cfg_interval = normalize_interval(get_param(task.params, "cfg_interval", None), DEFAULT_CFG_INTERVAL)

    image = inferencer.gen_image(
        target_shape,
        gen_context,
        cfg_text_precontext=cfg_text_ctx,
        cfg_img_precontext=cfg_img_ctx,
        cfg_text_scale=float(get_param(task.params, "cfg_text_scale", 4.0)),
        cfg_img_scale=float(get_param(task.params, "cfg_img_scale", 1.5)),
        cfg_interval=cfg_interval,
        timestep_shift=float(get_param(task.params, "timestep_shift", 3.0)),
        num_timesteps=int(get_param(task.params, "num_timesteps", 50)),
        enable_taylorseer=enable_taylorseer,
    )

    return image, None