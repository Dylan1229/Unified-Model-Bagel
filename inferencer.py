# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


@dataclass
class CachedContext:
    kv_lens: List[int]
    ropes: List[int]
    past_key_values: NaiveCache

    def clone(self) -> "CachedContext":
        return CachedContext(
            kv_lens=list(self.kv_lens),
            ropes=list(self.ropes),
            past_key_values=_clone_naive_cache(self.past_key_values),
        )

    def to_device_dict(self, device: torch.device) -> Dict[str, Any]:
        cache = _clone_naive_cache(self.past_key_values)
        _move_cache_inplace(cache, device)
        return {"kv_lens": list(self.kv_lens), "ropes": list(self.ropes), "past_key_values": cache}


@dataclass
class TextToImagePlan:
    image_shape: Tuple[int, int]
    generation_context: CachedContext
    cfg_text_context: CachedContext
    cfg_img_context: CachedContext
    diffusion_kwargs: Dict[str, Any]
    thinking_text: Optional[str] = None

    def contexts_for_device(self, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        return (
            self.generation_context.to_device_dict(device),
            self.cfg_text_context.to_device_dict(device),
            self.cfg_img_context.to_device_dict(device),
        )


def _clone_naive_cache(cache: NaiveCache) -> NaiveCache:
    cloned = NaiveCache(cache.num_layers)
    for layer_idx in range(cache.num_layers):
        key = cache.key_cache[layer_idx]
        value = cache.value_cache[layer_idx]
        if key is not None:
            cloned.key_cache[layer_idx] = key.detach().clone()
        if value is not None:
            cloned.value_cache[layer_idx] = value.detach().clone()
    return cloned


def _move_cache_inplace(cache: NaiveCache, device: torch.device) -> None:
    for layer_idx in range(cache.num_layers):
        key = cache.key_cache[layer_idx]
        value = cache.value_cache[layer_idx]
        if key is not None:
            cache.key_cache[layer_idx] = key.to(device, non_blocking=True)
        if value is not None:
            cache.value_cache[layer_idx] = value.to(device, non_blocking=True)


def _context_to_cached(context: Dict[str, Any]) -> CachedContext:
    kv_lens = [int(x) for x in context["kv_lens"]]
    ropes = [int(x) for x in context["ropes"]]
    cache = _clone_naive_cache(context["past_key_values"])
    _move_cache_inplace(cache, torch.device("cpu"))
    return CachedContext(kv_lens=kv_lens, ropes=ropes, past_key_values=cache)


VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def _ensure_vae_device(self, device: torch.device) -> None:
        try:
            param = next(self.vae_model.parameters())
        except StopIteration:
            return
        target_dtype = torch.float32 if device.type == "cuda" else param.dtype
        if param.device != device or param.dtype != target_dtype:
            self.vae_model = self.vae_model.to(device=device, dtype=target_dtype)
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )
        # Prefill stage
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            self._ensure_vae_device(torch.device("cuda", torch.cuda.current_device()))
            ## update vae
            torch.cuda.nvtx.range_push("vae_encoding")
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            param = next(self.vae_model.parameters(), None)
            if param is not None:
                generation_input["padded_images"] = generation_input["padded_images"].to(
                    device=param.device,
                    dtype=param.dtype,
                    non_blocking=True,
                )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
            torch.cuda.nvtx.range_pop()
        if vit:
            ## update vit
            torch.cuda.nvtx.range_push("vit_encoding")
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
            torch.cuda.nvtx.range_pop()
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        # print(cfg_renorm_type)
        # torch.cuda.nvtx.range_push("Prepare Latent & CFG")
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        device = torch.device("cuda", torch.cuda.current_device())
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        )
        generation_input = {
            key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in generation_input.items()
        }
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_text = {
            key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in generation_input_cfg_text.items()
        }

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )
        generation_input_cfg_img = {
            key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in generation_input_cfg_img.items()
        }
        # torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Diffusion Process")
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("VAE Decode")
        image = self.decode_image(unpacked_latent[0], image_shape)
        torch.cuda.nvtx.range_pop()
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        vae_device = None
        try:
            vae_device = next(self.vae_model.parameters()).device
        except StopIteration:
            pass
        if vae_device is not None and latent.device != vae_device:
            latent = latent.to(vae_device, non_blocking=True)

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        target_device = latent.device
        self._ensure_vae_device(target_device)
        param = next(self.vae_model.parameters(), None)
        print(f"[decode_image] latent_device={target_device} vae_device={param.device if param is not None else 'unknown'}")
        if param is not None and latent.dtype != param.dtype:
            latent = latent.to(param.dtype)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        if torch.cuda.is_available():
            self._ensure_vae_device(torch.device("cuda", torch.cuda.current_device()))

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):

            if think:
                torch.cuda.nvtx.range_push("Text prefill of system prompt (thinking mode)")
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)
                torch.cuda.nvtx.range_pop()

            for input_term in input_lists:

                if isinstance(input_term, str):
                    torch.cuda.nvtx.range_push("Text prefill of input text")
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                    torch.cuda.nvtx.range_pop()

                elif isinstance(input_term, Image.Image):
                    torch.cuda.nvtx.range_push("Input Image Processing")
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output) # Understanding: ViT, Generation: VAE
                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)
                    torch.cuda.nvtx.range_pop()

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                torch.cuda.nvtx.range_push("Text Decoding (Image Understanding Task)")
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)
                torch.cuda.nvtx.range_pop()

            else:
                if think:
                    torch.cuda.nvtx.range_push("Text Decoding (Thinking Mode in T2I Gen)")
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("Text Prefill Again (Thinking Mode in T2I Gen)")
                    gen_context = self.update_context_text(gen_text, gen_context)
                    torch.cuda.nvtx.range_pop()

                    output_list.append(gen_text)


                torch.cuda.nvtx.range_push("Image Generation")
                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )
                torch.cuda.nvtx.range_pop()
                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict

    @torch.no_grad()
    def prepare_text_to_image(
        self,
        prompt: str,
        think: bool = False,
        *,
        image_shape: Tuple[int, int],
        max_think_token_n: int = 512,
        do_sample: bool = False,
        text_temperature: float = 0.3,
        cfg_text_scale: float = 4.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: Tuple[float, float] = (0.4, 1.0),
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        enable_taylorseer: bool = False,
        device_id: Optional[int] = None,
    ) -> Tuple[TextToImagePlan, Optional[str]]:
        previous_device: Optional[int] = None
        if device_id is not None:
            previous_device = torch.cuda.current_device()
            if previous_device != device_id:
                torch.cuda.set_device(device_id)
        target_device = torch.device("cuda", torch.cuda.current_device())
        self._ensure_vae_device(target_device)
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        thinking_text: Optional[str] = None

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                gen_context = self.update_context_text(GEN_THINK_SYSTEM_PROMPT, gen_context)
                cfg_img_context = self.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img_context)

            cfg_text_context = deepcopy(gen_context)
            gen_context = self.update_context_text(prompt, gen_context)
            cfg_img_context = self.update_context_text(prompt, cfg_img_context)

            if think:
                thinking_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n,
                )
                gen_context = self.update_context_text(thinking_text, gen_context)

        plan = TextToImagePlan(
            image_shape=image_shape,
            generation_context=_context_to_cached(gen_context),
            cfg_text_context=_context_to_cached(cfg_text_context),
            cfg_img_context=_context_to_cached(cfg_img_context),
            diffusion_kwargs=dict(
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=tuple(cfg_interval),
                timestep_shift=timestep_shift,
                num_timesteps=num_timesteps,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                enable_taylorseer=enable_taylorseer,
            ),
            thinking_text=thinking_text,
        )
        if device_id is not None and previous_device is not None and previous_device != device_id:
            torch.cuda.set_device(previous_device)
        return plan, thinking_text

    @torch.no_grad()
    def render_text_to_image_plan(
        self,
        plan: TextToImagePlan,
        device_id: Optional[int] = None,
    ) -> Image.Image:
        if device_id is None:
            device_id = torch.cuda.current_device()
        previous_device = torch.cuda.current_device()
        if previous_device != device_id:
            torch.cuda.set_device(device_id)
        target_device = torch.device("cuda", device_id)
        self._ensure_vae_device(target_device)
        gen_context, cfg_text_context, cfg_img_context = plan.contexts_for_device(target_device)
        image = self.gen_image(
            plan.image_shape,
            gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            **plan.diffusion_kwargs,
        )
        if previous_device != device_id:
            torch.cuda.set_device(previous_device)
        return image
