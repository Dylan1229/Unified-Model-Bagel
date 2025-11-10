import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch  # noqa: E402

from data.transforms import ImageTransform  # noqa: E402
from data.data_utils import add_special_tokens  # noqa: E402
from inferencer import InterleaveInferencer  # noqa: E402
from modeling.autoencoder import load_ae  # noqa: E402
from modeling.bagel import (  # noqa: E402
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer  # noqa: E402


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert text to a safe filename prefix"""
    # Remove or replace invalid filename characters
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', text)
    # Replace spaces and multiple whitespaces with underscore
    text = re.sub(r'\s+', '_', text.strip())
    # Remove leading/trailing dots and underscores
    text = text.strip('._')
    # Limit length
    if len(text) > max_length:
        text = text[:max_length].rstrip('._')
    # Ensure it's not empty
    if not text:
        text = "prompt"
    return text


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path: str, max_mem_per_gpu: str = "80GiB"):
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
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(f"Device map: {device_map}")

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for module in same_device_modules:
            if module in device_map:
                device_map[module] = first_device
            else:
                device_map[module] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for module in same_device_modules:
            if module in device_map:
                device_map[module] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )
    model = model.eval()
    print("Model loaded successfully")

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch image understanding with BAGEL.")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT", help="Path to the model directory.")
    parser.add_argument("--prompt_pth", default="./data_profile/prompt/editing.csv", help="CSV file with 'prompt' and 'image' columns.")
    parser.add_argument("--output", default="./results/image_edit", help="Directory to store edited images and optional texts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_mem_per_gpu", default="80GiB", help="Max memory per GPU for dispatch.")
    parser.add_argument("--think", action="store_true", default=False, help="Enable think mode before image understanding.")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Enable sampling during text decoding.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature if do_sample is set.")
    parser.add_argument("--max_think_token_n", type=int, default=1000, help="Maximum number of thinking tokens.")
    return parser.parse_args()


def load_prompt_image_pairs(csv_path: Path) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header with 'prompt' and 'image' columns.")
        header_map = {name.lower(): name for name in reader.fieldnames}
        prompt_key = header_map.get("prompt")
        image_key = header_map.get("image")
        if prompt_key is None or image_key is None:
            raise ValueError(f"CSV header must contain 'prompt' and 'image' columns. Found: {reader.fieldnames}")

        base_dir = csv_path.parent
        for row in reader:
            prompt = (row.get(prompt_key) or "").strip()
            image_rel = (row.get(image_key) or "").strip()
            if not prompt or not image_rel:
                continue
            # image_path = (base_dir / image_rel).expanduser().resolve()
            image_path = Path(image_rel).expanduser().resolve()
            pairs.append((prompt, image_path))
    return pairs


def main() -> None:
    args = parse_args()

    setup_seed(args.seed)

    csv_path = Path(args.prompt_pth).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    pairs = load_prompt_image_pairs(csv_path)
    if not pairs:
        raise ValueError(f"No valid prompt/image pairs found in {csv_path}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.nvtx.range_push("Model Loading")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        args.model_path,
        max_mem_per_gpu=args.max_mem_per_gpu,
    )
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Inferencer Init")
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    torch.cuda.nvtx.range_pop()

    inference_hyper = dict(
        max_think_token_n=args.max_think_token_n,
        do_sample=args.do_sample,
        text_temperature=args.temperature,
        understanding_output=True,
    )

    for idx, (prompt, image_path) in enumerate(pairs):
        if not image_path.is_file():
            print(f"Warning: image not found for row {idx}: {image_path}")
            continue

        print(f"[{idx + 1}/{len(pairs)}] Understanding image {image_path} with prompt: {prompt}")
        with Image.open(image_path) as pil_image:
            pil_image = pil_image.convert("RGB")
            output = inferencer(image=pil_image, text=prompt, **inference_hyper)

        text_output = output.get("text")
        if not text_output:
            print(f"Warning: no textual output produced for row {idx}")
            continue
        
        # Generate unique filename using prompt prefix and timestamp
        prompt_prefix = sanitize_filename(prompt)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        filename_base = f"{prompt_prefix}_{timestamp}"
        
        text_out_path = output_dir / f"{filename_base}.txt"
        text_out_path.write_text(text_output, encoding="utf-8")
        print(f"Saved text output to {text_out_path}")


if __name__ == "__main__":
    main()
