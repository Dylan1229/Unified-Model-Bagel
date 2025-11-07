from __future__ import annotations

import queue
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from inferencer import InterleaveInferencer, TextToImagePlan


class ParallelMode(str, Enum):
    DATA_PARALLEL = "data_parallel"
    STAGED_DIFFUSION = "staged_diffusion"


class DiffusionTaskKind(str, Enum):
    TEXT_TO_IMAGE = "text2image"
    IMAGE_EDITING = "image_editing"


@dataclass
class TextToImageRequest:
    task_index: int
    task: Any
    prompt: str
    think: bool
    plan_kwargs: Dict[str, Any]
    kind: str = DiffusionTaskKind.TEXT_TO_IMAGE.value
    image_path: Optional[Path] = None


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
    def __init__(
        self,
        inferencer: InterleaveInferencer,
        diffusion_device: int,
        diffusion_batch_size: int = 1,
    ) -> None:
        self._inferencer = inferencer
        self._diffusion_device = diffusion_device
        self._completed: List[TextToImageResult] = []
        self._batch_size = max(1, diffusion_batch_size)
        self._pending_jobs: Dict[DiffusionSignature, List[_PreparedDiffusionJob]] = defaultdict(list)

    def submit_text_to_image(self, request: TextToImageRequest) -> None:
        job = self._prepare_job(request)
        signature = job.plan.diffusion_signature()
        bucket = self._pending_jobs[signature]
        bucket.append(job)
        if len(bucket) >= self._batch_size:
            self._process_signature_queue(signature, force=False)

    def poll(self) -> List[TextToImageResult]:
        if not self._completed:
            return []
        results = self._completed
        self._completed = []
        return results

    def wait_all(self) -> List[TextToImageResult]:
        self._run_pending_batch(force=True)
        return self.poll()

    def _prepare_job(self, request: TextToImageRequest) -> _PreparedDiffusionJob:
        if request.kind == DiffusionTaskKind.IMAGE_EDITING.value:
            if request.image_path is None:
                raise ValueError("image_path is required for image editing tasks.")
            with Image.open(request.image_path) as image:
                image = image.convert("RGB")
                plan, thinking_text = self._inferencer.prepare_image_editing(
                    image=image,
                    prompt=request.prompt,
                    think=request.think,
                    device_id=self._diffusion_device,
                    **request.plan_kwargs,
                )
        else:
            plan, thinking_text = self._inferencer.prepare_text_to_image(
                request.prompt,
                think=request.think,
                device_id=self._diffusion_device,
                **request.plan_kwargs,
            )
        return _PreparedDiffusionJob(request=request, plan=plan, thinking_text=thinking_text)

    def _run_pending_batch(self, force: bool = False) -> None:
        if not self._pending_jobs:
            return
        if force:
            signatures = list(self._pending_jobs.keys())
            for signature in signatures:
                self._process_signature_queue(signature, force=True)

    def _process_signature_queue(self, signature: DiffusionSignature, force: bool) -> None:
        queue = self._pending_jobs.get(signature)
        if not queue:
            return
        while queue and (force or len(queue) >= self._batch_size):
            current = min(self._batch_size, len(queue))
            jobs = queue[:current]
            del queue[:current]
            self._execute_batch(jobs)
        if not queue:
            del self._pending_jobs[signature]

    def _execute_batch(self, jobs: List[_PreparedDiffusionJob]) -> None:
        images = self._inferencer.render_text_to_image_plan_batch(
            [job.plan for job in jobs],
            device_id=self._diffusion_device,
        )
        for job, image in zip(jobs, images):
            self._completed.append(
                TextToImageResult(
                    task_index=job.request.task_index,
                    task=job.request.task,
                    image=image,
                    thinking_text=job.thinking_text,
                )
            )


@dataclass
class _DecodeWorker:
    device_id: int
    inferencer: InterleaveInferencer
    executor: ThreadPoolExecutor


@dataclass
class _PreparedPlan:
    plan: TextToImagePlan
    thinking_text: Optional[str]


@dataclass
class _PreparedDiffusionJob:
    request: TextToImageRequest
    plan: TextToImagePlan
    thinking_text: Optional[str]


DiffusionSignature = Tuple[Tuple[str, Any], ...]


class StagedDiffusionScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        diffusion_device: int,
        decode_devices: List[int],
        inferencer_factory: Callable[[int], InterleaveInferencer],
        diffusion_batch_size: int = 1,
    ) -> None:
        if not decode_devices:
            raise ValueError("decode_devices must contain at least one GPU index for staged diffusion.")
        if diffusion_device in decode_devices:
            raise ValueError("diffusion_device must be distinct from decode_devices.")

        self._diffusion_device = diffusion_device
        self._decode_workers = [
            _DecodeWorker(
                device_id=device_id,
                inferencer=inferencer_factory(device_id),
                executor=ThreadPoolExecutor(max_workers=1),
            )
            for device_id in decode_devices
        ]
        self._diffusion_inferencer = inferencer_factory(diffusion_device)
        self._diffusion_executor = ThreadPoolExecutor(max_workers=1)

        self._worker_lock = Lock()
        self._next_worker_index = 0

        self._pending_lock = Lock()
        self._pending_jobs = 0
        self._results: "queue.Queue[TextToImageResult]" = queue.Queue()
        self._diffusion_queue_lock = Lock()
        self._pending_diffusion_jobs: Dict[DiffusionSignature, List[_PreparedDiffusionJob]] = defaultdict(list)
        self._diffusion_batch_size = max(1, diffusion_batch_size)

    def submit_text_to_image(self, request: TextToImageRequest) -> None:
        worker = self._select_worker()
        with self._pending_lock:
            self._pending_jobs += 1

        future = worker.executor.submit(self._run_prepare, worker, request)
        future.add_done_callback(lambda fut, req=request: self._on_plan_ready(req, fut))

    def poll(self) -> List[TextToImageResult]:
        results: List[TextToImageResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        return results

    def wait_all(self) -> List[TextToImageResult]:
        self._flush_diffusion_jobs()
        results: List[TextToImageResult] = []
        while True:
            try:
                result = self._results.get(timeout=0.1)
            except queue.Empty:
                with self._pending_lock:
                    if self._pending_jobs == 0:
                        break
                continue
            results.append(result)

        results.extend(self.poll())
        return results

    def shutdown(self) -> None:
        for worker in self._decode_workers:
            worker.executor.shutdown(wait=True, cancel_futures=True)
        self._diffusion_executor.shutdown(wait=True, cancel_futures=True)

    def _select_worker(self) -> _DecodeWorker:
        with self._worker_lock:
            worker = self._decode_workers[self._next_worker_index]
            self._next_worker_index = (self._next_worker_index + 1) % len(self._decode_workers)
        return worker

    @staticmethod
    def _run_prepare(worker: _DecodeWorker, request: TextToImageRequest) -> _PreparedPlan:
        if request.kind == DiffusionTaskKind.IMAGE_EDITING.value:
            if request.image_path is None:
                raise ValueError("image_path is required for image editing tasks.")
            with Image.open(request.image_path) as image:
                image = image.convert("RGB")
                plan, thinking = worker.inferencer.prepare_image_editing(
                    image=image,
                    prompt=request.prompt,
                    think=request.think,
                    device_id=worker.device_id,
                    **request.plan_kwargs,
                )
        else:
            plan, thinking = worker.inferencer.prepare_text_to_image(
                request.prompt,
                think=request.think,
                device_id=worker.device_id,
                **request.plan_kwargs,
            )
        return _PreparedPlan(plan=plan, thinking_text=thinking)

    def _on_plan_ready(self, request: TextToImageRequest, future: Future[_PreparedPlan]) -> None:
        try:
            prepared = future.result()
        except BaseException as exc:  # noqa: BLE001
            self._enqueue_result(
                TextToImageResult(
                    task_index=request.task_index,
                    task=request.task,
                    error=exc,
                )
            )
            return

        job = _PreparedDiffusionJob(request=request, plan=prepared.plan, thinking_text=prepared.thinking_text)
        self._queue_diffusion_job(job)

    def _queue_diffusion_job(self, job: _PreparedDiffusionJob) -> None:
        signature = job.plan.diffusion_signature()
        with self._diffusion_queue_lock:
            bucket = self._pending_diffusion_jobs[signature]
            bucket.append(job)
            self._submit_diffusion_jobs_locked(signature, force=False)

    def _flush_diffusion_jobs(self) -> None:
        with self._diffusion_queue_lock:
            self._submit_diffusion_jobs_locked(signature=None, force=True)

    def _submit_diffusion_jobs_locked(
        self,
        signature: Optional[DiffusionSignature],
        force: bool,
    ) -> None:
        signatures = [signature] if signature is not None else list(self._pending_diffusion_jobs.keys())
        for sig in signatures:
            queue = self._pending_diffusion_jobs.get(sig)
            if not queue:
                continue
            while queue and (force or len(queue) >= self._diffusion_batch_size):
                current = min(self._diffusion_batch_size, len(queue))
                batch = queue[:current]
                del queue[:current]
                future = self._diffusion_executor.submit(self._run_diffusion_batch, batch)
                future.add_done_callback(lambda fut, jobs=batch: self._on_diffusion_batch_done(jobs, fut))
            if not queue:
                self._pending_diffusion_jobs.pop(sig, None)

    def _on_diffusion_batch_done(self, jobs: List[_PreparedDiffusionJob], future: Future[List[Image.Image]]) -> None:
        try:
            images = future.result()
        except BaseException as exc:  # noqa: BLE001
            for job in jobs:
                self._enqueue_result(
                    TextToImageResult(
                        task_index=job.request.task_index,
                        task=job.request.task,
                        thinking_text=job.thinking_text,
                        error=exc,
                    )
                )
            return

        for job, image in zip(jobs, images):
            self._enqueue_result(
                TextToImageResult(
                    task_index=job.request.task_index,
                    task=job.request.task,
                    image=image,
                    thinking_text=job.thinking_text,
                )
            )

    def _enqueue_result(self, result: TextToImageResult) -> None:
        self._results.put(result)
        with self._pending_lock:
            self._pending_jobs -= 1

    def _run_diffusion_batch(self, jobs: List[_PreparedDiffusionJob]) -> List[Image.Image]:
        return self._diffusion_inferencer.render_text_to_image_plan_batch(
            [job.plan for job in jobs],
            device_id=self._diffusion_device,
        )


def build_scheduler(
    mode: ParallelMode,
    *,
    inferencer_factory: Callable[[int], InterleaveInferencer],
    diffusion_device: int,
    decode_devices: Optional[List[int]] = None,
    diffusion_batch_size: int = 1,
) -> BaseScheduler:
    if mode == ParallelMode.DATA_PARALLEL:
        return DataParallelScheduler(
            inferencer_factory(diffusion_device),
            diffusion_device,
            diffusion_batch_size=diffusion_batch_size,
        )

    if mode == ParallelMode.STAGED_DIFFUSION:
        if decode_devices is None or not decode_devices:
            raise ValueError("decode_devices must be provided for staged diffusion mode.")
        return StagedDiffusionScheduler(
            diffusion_device=diffusion_device,
            decode_devices=decode_devices,
            inferencer_factory=inferencer_factory,
            diffusion_batch_size=diffusion_batch_size,
        )

    raise ValueError(f"Unsupported parallel mode: {mode}")
