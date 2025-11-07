from __future__ import annotations

import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from inferencer import InterleaveInferencer, TextToImagePlan
from scripts.inference_serving.diffusion_batcher import DiffusionBatchExecutor


class ParallelMode(str, Enum):
    DATA_PARALLEL = "data_parallel"
    STAGED_DIFFUSION = "staged_diffusion"


@dataclass
class TextToImageRequest:
    task_index: int
    task: Any
    build_plan: Callable[[InterleaveInferencer, int], Tuple[TextToImagePlan, Optional[str]]]


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


@dataclass
class _PendingDiffusion:
    request: TextToImageRequest
    thinking_text: Optional[str]
    future: Future


class DataParallelScheduler(BaseScheduler):
    def __init__(
        self,
        inferencer: InterleaveInferencer,
        diffusion_device: int,
        batch_executor: DiffusionBatchExecutor,
    ) -> None:
        self._inferencer = inferencer
        self._diffusion_device = diffusion_device
        self._batch_executor = batch_executor
        self._completed: List[TextToImageResult] = []
        self._pending: List[_PendingDiffusion] = []

    def submit_text_to_image(self, request: TextToImageRequest) -> None:
        try:
            plan, thinking_text = request.build_plan(self._inferencer, self._diffusion_device)
        except BaseException as exc:  # noqa: BLE001
            self._completed.append(
                TextToImageResult(
                    task_index=request.task_index,
                    task=request.task,
                    error=exc,
                )
            )
            return
        future = self._batch_executor.submit(plan)
        self._pending.append(_PendingDiffusion(request=request, thinking_text=thinking_text, future=future))

    def poll(self) -> List[TextToImageResult]:
        results: List[TextToImageResult] = []
        remaining: List[_PendingDiffusion] = []
        for pending in self._pending:
            if pending.future.done():
                try:
                    image = pending.future.result()
                    results.append(
                        TextToImageResult(
                            task_index=pending.request.task_index,
                            task=pending.request.task,
                            image=image,
                            thinking_text=pending.thinking_text,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001
                    results.append(
                        TextToImageResult(
                            task_index=pending.request.task_index,
                            task=pending.request.task,
                            thinking_text=pending.thinking_text,
                            error=exc,
                        )
                    )
            else:
                remaining.append(pending)
        self._pending = remaining

        if self._completed:
            results.extend(self._completed)
            self._completed = []
        return results

    def wait_all(self) -> List[TextToImageResult]:
        self._batch_executor.flush()
        results: List[TextToImageResult] = []
        while self._pending or self._completed:
            polled = self.poll()
            if polled:
                results.extend(polled)
            if self._pending:
                time.sleep(0.01)
        results.extend(self.poll())
        return results

    def shutdown(self) -> None:
        self._batch_executor.shutdown()


@dataclass
class _DecodeWorker:
    device_id: int
    inferencer: InterleaveInferencer
    executor: ThreadPoolExecutor


@dataclass
class _PreparedPlan:
    plan: TextToImagePlan
    thinking_text: Optional[str]


class StagedDiffusionScheduler(BaseScheduler):
    def __init__(
        self,
        *,
        diffusion_device: int,
        decode_devices: List[int],
        inferencer_factory: Callable[[int], InterleaveInferencer],
        batch_executor: DiffusionBatchExecutor,
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
        self._batch_executor = batch_executor

        self._worker_lock = Lock()
        self._next_worker_index = 0

        self._pending_lock = Lock()
        self._pending_jobs = 0
        self._results: "queue.Queue[TextToImageResult]" = queue.Queue()

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
        self._batch_executor.flush()
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
        self._batch_executor.shutdown()

    def _select_worker(self) -> _DecodeWorker:
        with self._worker_lock:
            worker = self._decode_workers[self._next_worker_index]
            self._next_worker_index = (self._next_worker_index + 1) % len(self._decode_workers)
        return worker

    @staticmethod
    def _run_prepare(worker: _DecodeWorker, request: TextToImageRequest) -> _PreparedPlan:
        plan, thinking = request.build_plan(worker.inferencer, worker.device_id)
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

        diffusion_future = self._batch_executor.submit(prepared.plan)
        diffusion_future.add_done_callback(
            lambda fut, req=request, thinking=prepared.thinking_text: self._on_diffusion_ready(req, thinking, fut)
        )

    def _on_diffusion_ready(
        self,
        request: TextToImageRequest,
        thinking_text: Optional[str],
        future: Future[Image.Image],
    ) -> None:
        try:
            image = future.result()
        except BaseException as exc:  # noqa: BLE001
            self._enqueue_result(
                TextToImageResult(task_index=request.task_index, task=request.task, thinking_text=thinking_text, error=exc)
            )
            return

        self._enqueue_result(
            TextToImageResult(
                task_index=request.task_index,
                task=request.task,
                image=image,
                thinking_text=thinking_text,
            )
        )

    def _enqueue_result(self, result: TextToImageResult) -> None:
        self._results.put(result)
        with self._pending_lock:
            self._pending_jobs -= 1


def build_scheduler(
    mode: ParallelMode,
    *,
    inferencer_factory: Callable[[int], InterleaveInferencer],
    diffusion_device: int,
    default_batch_size: int = 1,
    batch_size_resolver: Optional[Callable[[TextToImagePlan], int]] = None,
    decode_devices: Optional[List[int]] = None,
) -> BaseScheduler:
    if mode == ParallelMode.DATA_PARALLEL:
        inferencer = inferencer_factory(diffusion_device)
        batch_executor = DiffusionBatchExecutor(
            inferencer=inferencer,
            device_id=diffusion_device,
            batch_size=default_batch_size,
            async_dispatch=False,
            batch_size_resolver=batch_size_resolver,
        )
        return DataParallelScheduler(inferencer, diffusion_device, batch_executor)

    if mode == ParallelMode.STAGED_DIFFUSION:
        if decode_devices is None or not decode_devices:
            raise ValueError("decode_devices must be provided for staged diffusion mode.")
        batch_executor = DiffusionBatchExecutor(
            inferencer=inferencer_factory(diffusion_device),
            device_id=diffusion_device,
            batch_size=default_batch_size,
            async_dispatch=True,
            batch_size_resolver=batch_size_resolver,
        )
        return StagedDiffusionScheduler(
            diffusion_device=diffusion_device,
            decode_devices=decode_devices,
            inferencer_factory=inferencer_factory,
            batch_executor=batch_executor,
        )

    raise ValueError(f"Unsupported parallel mode: {mode}")
