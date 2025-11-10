from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from inferencer import InterleaveInferencer
from text_to_image import TextToImageEngine, TextToImagePlan

# Types of parallel scheduling modes.
# DP: Just data parallelism across all available GPUs.
# MP: Model parallelism across GPUs. Put diffusion stages in some gpus and put the rest of the stages in other gpus.

#
class ParallelMode(str, Enum):
    DP = "data_parallel"
    MP = "model_parallel"


@dataclass
class TextToImageRequest:
    task_index: int
    task: Any
    prompt: str
    think: bool
    plan_kwargs: Dict[str, Any]


@dataclass
class TextToImageResult:
    task_index: int
    task: Any
    image: Optional[Image.Image] = None
    thinking_text: Optional[str] = None
    error: Optional[BaseException] = None


class BaseScheduler:
    def submit_text_to_image(self, request: TextToImageRequest) -> None:
        raise NotImplementedError

    def poll(self) -> List[TextToImageResult]:
        raise NotImplementedError

    def wait_all(self) -> List[TextToImageResult]:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class DataParallelScheduler(BaseScheduler):
    def __init__(self, inferencer: InterleaveInferencer, diffusion_device: int) -> None:
        self._inferencer = inferencer
        self._text_to_image = inferencer.text_to_image
        self._diffusion_device = diffusion_device
        self._completed: List[TextToImageResult] = []

    def submit_text_to_image(self, request: TextToImageRequest) -> None:
        plan, thinking_text = self._text_to_image.prepare_text_to_image(
            request.prompt,
            think=request.think,
            device_id=self._diffusion_device,
            **request.plan_kwargs,
        )
        image = self._text_to_image.render_text_to_image_plan(plan, device_id=self._diffusion_device)
        self._completed.append(
            TextToImageResult(
                task_index=request.task_index,
                task=request.task,
                image=image,
                thinking_text=thinking_text,
            )
        )

    def poll(self) -> List[TextToImageResult]:
        if not self._completed:
            return []
        results = self._completed
        self._completed = []
        return results

    def wait_all(self) -> List[TextToImageResult]:
        return self.poll()


@dataclass
class _DecodeWorker:
    device_id: int
    text_to_image: TextToImageEngine
    executor: ThreadPoolExecutor


@dataclass
class _PreparedPlan:
    plan: TextToImagePlan
    thinking_text: Optional[str]

def build_scheduler(
    mode: ParallelMode,
    *,
    inferencer_factory: Callable[[int], InterleaveInferencer],
    diffusion_device: int,
    decode_devices: Optional[List[int]] = None,
) -> BaseScheduler:
    if mode == ParallelMode.DP:
        return DataParallelScheduler(inferencer_factory(diffusion_device), diffusion_device)

    if mode == ParallelMode.MP:
        raise NotImplementedError("Model parallel scheduler is not implemented yet.")

    raise ValueError(f"Unsupported parallel mode: {mode}")
