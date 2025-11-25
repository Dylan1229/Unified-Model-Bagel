import argparse
import sys
import time
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from PIL import Image

# Disable TQDM globally to prevent stdout deadlocks
os.environ["TQDM_DISABLE"] = "1" 

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inferencer import InterleaveInferencer, GEN_THINK_SYSTEM_PROMPT
from inference.utils.utils import TaskSpec, ensure_path, load_model, load_tasks, setup_seed

TEXT_TO_IMAGE_KINDS = {"text2image", "text-to-image", "txt2img", "txt-to-img"}
DEFAULT_CFG_INTERVAL = (0.4, 1.0)

# A100 Optimization
torch.set_float32_matmul_precision('high')

@dataclass
class StageState:
    task: TaskSpec
    gen_context: Dict[str, Any] = field(default_factory=dict)
    cfg_text_context: Dict[str, Any] = field(default_factory=dict)
    cfg_img_context: Dict[str, Any] = field(default_factory=dict)
    image_shape: Tuple[int, int] = (1024, 1024)
    params: Dict[str, Any] = field(default_factory=dict)
    think_text: Optional[str] = None
    image: Optional[Image.Image] = None
    text_ready: bool = False

    def to_device(self, device: int):
        def _move(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    if v.device.index != device:
                        new_d[k] = v.to(device, non_blocking=True)
                    else:
                        new_d[k] = v
                else:
                    new_d[k] = v
            return new_d

        self.gen_context = _move(self.gen_context)
        self.cfg_text_context = _move(self.cfg_text_context)
        self.cfg_img_context = _move(self.cfg_img_context)

def _get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    if params is None: return default
    return params.get(key, default)

def _normalize_interval(raw: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if raw is None: return fallback
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    return fallback

# --- Worker Functions ---

def text_worker_proc(
    in_queue: mp.Queue,
    inter_queue: mp.Queue,
    result_queue: mp.Queue,
    load_event: mp.Event,
    shutdown_event: mp.Event,  # <--- Added Shutdown Signal
    model_path: str,
    device_id: int,
    max_mem_per_gpu: str,
    seed: int
):
    setup_seed(seed)
    torch.cuda.set_device(device_id)
    torch.set_float32_matmul_precision('high')
    
    print(f"[TextWorker] Loading model (Limit: {max_mem_per_gpu})...")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        model_path,
        max_mem_per_gpu=max_mem_per_gpu,
        device_ids=[device_id],
    )
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    
    print(f"[TextWorker] Load Complete. Signaling ImageWorker...")
    load_event.set() 
    
    while True:
        payload = in_queue.get()
        if payload is None:
            inter_queue.put(None) 
            print("[TextWorker] All tasks processed. Waiting for shutdown signal...")
            # [FIX] Wait for event instead of sleeping forever
            shutdown_event.wait()
            break # Graceful exit
        
        task, args_dict = payload
        try:
            state = prepare_state_local(inferencer, task, args_dict['default_shape'])
            if state.task.think:
                print(f"[{state.task.task_id}] TextWorker: Generating plan...")
                torch.cuda.nvtx.range_push(f"text_decode:{state.task.task_id}")
                text = inferencer.gen_text(
                    state.gen_context,
                    do_sample=bool(_get_param(state.params, "do_sample", args_dict['do_sample'])),
                    temperature=_get_param(state.params, "text_temperature", args_dict['text_temperature']),
                    max_length=int(_get_param(state.params, "max_think_token_n", args_dict['max_think_token_n'])),
                )
                torch.cuda.nvtx.range_pop()
                state.gen_context = inferencer.update_context_text(text, state.gen_context)
                state.cfg_img_context = inferencer.update_context_text(text, state.cfg_img_context)
                state.think_text = text
                state.text_ready = True
                result_queue.put(('text', state.task.task_id, text, state.task.output_text))

            inter_queue.put((state, args_dict))
        except Exception as e:
            print(f"[TextWorker] Error on {task.task_id}: {e}")
            import traceback; traceback.print_exc()

def image_worker_proc(
    inter_queue: mp.Queue,
    result_queue: mp.Queue,
    load_event: mp.Event,
    shutdown_event: mp.Event, # <--- Added Shutdown Signal
    model_path: str,
    device_id: int,
    max_mem_per_gpu: str,
    seed: int
):
    setup_seed(seed)
    torch.cuda.set_device(device_id)
    torch.set_float32_matmul_precision('high')

    print(f"[ImageWorker] Waiting for TextWorker to finish loading...")
    load_event.wait() 
    
    print(f"[ImageWorker] Loading model (Limit: {max_mem_per_gpu})...")
    model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = load_model(
        model_path,
        max_mem_per_gpu=max_mem_per_gpu,
        device_ids=[device_id],
    )
    inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)
    print(f"[ImageWorker] Ready.")

    while True:
        payload = inter_queue.get()
        if payload is None:
            print("[ImageWorker] Finished all tasks. Waiting for shutdown signal...")
            # [FIX] Wait for event instead of sleeping forever
            shutdown_event.wait()
            break # Graceful exit

        state, args_dict = payload
        try:
            state.to_device(device_id)
            print(f"[{state.task.task_id}] ImageWorker: Generating image...")
            cfg_interval = _normalize_interval(_get_param(state.params, "cfg_interval", args_dict['cfg_interval']), DEFAULT_CFG_INTERVAL)
            
            torch.cuda.nvtx.range_push(f"diffusion:{state.task.task_id}")
            image = inferencer.gen_image(
                state.image_shape,
                state.gen_context,
                cfg_text_precontext=state.cfg_text_context,
                cfg_img_precontext=state.cfg_img_context,
                cfg_text_scale=float(_get_param(state.params, "cfg_text_scale", args_dict['cfg_text_scale'])),
                cfg_img_scale=float(_get_param(state.params, "cfg_img_scale", args_dict['cfg_img_scale'])),
                cfg_interval=cfg_interval,
                timestep_shift=float(_get_param(state.params, "timestep_shift", args_dict['timestep_shift'])),
                num_timesteps=int(_get_param(state.params, "num_timesteps", args_dict['num_timesteps'])),
                cfg_renorm_min=float(_get_param(state.params, "cfg_renorm_min", args_dict['cfg_renorm_min'])),
                cfg_renorm_type=_get_param(state.params, "cfg_renorm_type", args_dict['cfg_renorm_type']),
                enable_taylorseer=args_dict['enable_taylorseer'],
            )
            torch.cuda.nvtx.range_pop()
            result_queue.put(('image', state.task.task_id, image, state.task.output_image))
        except Exception as e:
            print(f"[ImageWorker] Error on {state.task.task_id}: {e}")
            import traceback; traceback.print_exc()

def prepare_state_local(inferencer, task, default_shape):
    base_context = inferencer.init_gen_context()
    cfg_text = deepcopy(base_context)
    cfg_img = deepcopy(base_context)
    prompt_text = task.prompt or ""
    if task.think:
        base_context = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, base_context)
        cfg_img = inferencer.update_context_text(GEN_THINK_SYSTEM_PROMPT, cfg_img)
    base_context = inferencer.update_context_text(prompt_text, base_context)
    cfg_text = deepcopy(base_context)
    cfg_img = inferencer.update_context_text(prompt_text, cfg_img)
    image_shape = task.image_shape or default_shape
    params = task.params or {}
    return StageState(task=task, gen_context=base_context, cfg_text_context=cfg_text, 
                      cfg_img_context=cfg_img, image_shape=image_shape, params=params)

def overlap_pipeline_mp(tasks, args, output_dir, image_device, text_device):
    task_queue = mp.Queue()
    inter_queue = mp.Queue()
    result_queue = mp.Queue()
    load_event = mp.Event() 
    shutdown_event = mp.Event() # [FIX] Create shutdown event

    args_dict = vars(args)
    ctx = mp.get_context('spawn')
    
    p_text = ctx.Process(
        target=text_worker_proc,
        args=(task_queue, inter_queue, result_queue, load_event, shutdown_event, args.model_path, text_device, args.max_mem_per_gpu, args.seed),
        daemon=True 
    )
    p_image = ctx.Process(
        target=image_worker_proc,
        args=(inter_queue, result_queue, load_event, shutdown_event, args.model_path, image_device, args.max_mem_per_gpu, args.seed),
        daemon=True
    )

    p_text.start()
    p_image.start()

    print(f"[Main] Submitting {len(tasks)} tasks...")
    for task in tasks:
        task_queue.put((task, args_dict))
    task_queue.put(None) 

    completed_images = 0
    total_tasks = len(tasks)

    try:
        while completed_images < total_tasks:
            msg = result_queue.get()
            kind, task_id = msg[0], msg[1]
            if kind == 'text':
                _, _, text_content, rel_path = msg
                out_path = ensure_path(rel_path, f"{task_id}_plan", output_dir, ".txt")
                out_path.write_text(text_content, encoding="utf-8")
                print(f"[{task_id}] Saved Plan Text.")
            elif kind == 'image':
                _, _, img_obj, rel_path = msg
                out_path = ensure_path(rel_path, task_id, output_dir, ".png")
                img_obj.save(out_path)
                print(f"[{task_id}] Saved Image.")
                completed_images += 1
    finally:
        print("[Main] All tasks completed. Signaling shutdown...")
        # [FIX] Signal workers to exit gracefully
        shutdown_event.set()
        
        # Wait for workers to clean up
        p_text.join(timeout=10)
        p_image.join(timeout=10)
        
        # Close queues to prevent leaks
        task_queue.close(); task_queue.join_thread()
        inter_queue.close(); inter_queue.join_thread()
        result_queue.close(); result_queue.join_thread()

        # Last resort kill if they are still stuck
        if p_text.is_alive(): p_text.terminate()
        if p_image.is_alive(): p_image.terminate()

def parse_args():
    parser = argparse.ArgumentParser(description="Test text/image stage overlap for BAGEL.")
    parser.add_argument("--model_path", default="./models/BAGEL-7B-MoT")
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("./results/overlap"))
    parser.add_argument("--max_mem_per_gpu", default="80GiB")
    parser.add_argument("--default_shape", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image_device", type=int, default=None)
    parser.add_argument("--text_device", type=int, default=None)
    parser.add_argument("--enable_taylorseer", action="store_true")
    parser.add_argument("--text_temperature", type=float, default=0.3)
    parser.add_argument("--max_think_token_n", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--cfg_interval", type=float, nargs=2, default=DEFAULT_CFG_INTERVAL)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", default="global")
    return parser.parse_args()

def main():
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    args = parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    
    import os
    os.environ["TQDM_DISABLE"] = "1"

    base_device = args.device
    image_device = args.image_device if args.image_device is not None else base_device
    text_device = args.text_device if args.text_device is not None else base_device
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in load_tasks(args.tasks, output_dir) if task.kind in TEXT_TO_IMAGE_KINDS]
    if len(tasks) < 1: raise RuntimeError("No text-to-image tasks found.")

    start = time.perf_counter()
    overlap_pipeline_mp(tasks, args, output_dir, image_device, text_device)
    duration = time.perf_counter() - start
    print(f"Completed overlap run for {len(tasks)} task(s) in {duration:.2f}s")

if __name__ == "__main__":
    main()