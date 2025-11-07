from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple

from inferencer import InterleaveInferencer, TextToImagePlan


_Signature = Tuple[Tuple[str, Any], ...]


@dataclass
class _BatchItem:
    plan: TextToImagePlan
    future: Future


class DiffusionBatchExecutor:
    def __init__(
        self,
        inferencer: InterleaveInferencer,
        device_id: int,
        *,
        batch_size: int = 1,
        async_dispatch: bool = False,
        batch_size_resolver: Optional[Callable[[TextToImagePlan], int]] = None,
    ) -> None:
        self._inferencer = inferencer
        self._device_id = device_id
        self._batch_size = max(1, batch_size)
        self._async_dispatch = async_dispatch
        self._queues: DefaultDict[_Signature, List[_BatchItem]] = defaultdict(list)
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1) if async_dispatch else None
        self._shutdown = False
        self._batch_size_resolver = batch_size_resolver

    def submit(self, plan: TextToImagePlan) -> Future:
        future: Future = Future()
        effective_batch_size = self._resolve_batch_size(plan)
        signature = self._build_signature(plan, effective_batch_size)
        batches: List[List[_BatchItem]]
        with self._lock:
            if self._shutdown:
                future.set_exception(RuntimeError("Diffusion batch executor has been shut down."))
                return future
            queue = self._queues[signature]
            queue.append(_BatchItem(plan=plan, future=future))
            batches = self._pop_batches(queue, effective_batch_size, drain_remaining=False)
        for batch in batches:
            self._dispatch(batch)
        return future

    def flush(self) -> None:
        pending: List[List[_BatchItem]] = []
        with self._lock:
            for signature in list(self._queues.keys()):
                queue = self._queues[signature]
                batch_size = self._batch_size_from_signature(signature)
                pending.extend(self._pop_batches(queue, batch_size, drain_remaining=True))
                if not queue:
                    del self._queues[signature]
        for batch in pending:
            self._dispatch(batch)

    def shutdown(self) -> None:
        self.flush()
        with self._lock:
            self._shutdown = True
            leftover: List[_BatchItem] = []
            for queue in self._queues.values():
                leftover.extend(queue)
                queue.clear()
        for item in leftover:
            if not item.future.done():
                item.future.set_exception(RuntimeError("Diffusion batch executor terminated before completion."))
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _pop_batches(self, queue: List[_BatchItem], batch_size: int, *, drain_remaining: bool) -> List[List[_BatchItem]]:
        batches: List[List[_BatchItem]] = []
        while len(queue) >= batch_size:
            batch = queue[:batch_size]
            del queue[:batch_size]
            batches.append(batch)
        if drain_remaining and queue:
            batches.append(list(queue))
            queue.clear()
        return batches

    def _dispatch(self, batch: List[_BatchItem]) -> None:
        if not batch:
            return
        if self._async_dispatch:
            assert self._executor is not None
            self._executor.submit(self._run_batch, batch)
        else:
            self._run_batch(batch)

    def _run_batch(self, batch: Sequence[_BatchItem]) -> None:
        plans = [item.plan for item in batch]
        try:
            images = self._inferencer.render_text_to_image_plans(plans, device_id=self._device_id)
        except BaseException as exc:  # noqa: BLE001
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        for item, image in zip(batch, images):
            if not item.future.done():
                item.future.set_result(image)

    def _resolve_batch_size(self, plan: TextToImagePlan) -> int:
        if self._batch_size_resolver is not None:
            resolved = self._batch_size_resolver(plan)
            if resolved and resolved > 0:
                return resolved
        return self._batch_size

    @staticmethod
    def _build_signature(plan: TextToImagePlan, batch_size: int) -> _Signature:
        items = list(plan.diffusion_kwargs.items())
        items.append(("batch_size", batch_size))
        return tuple(sorted(items))

    @staticmethod
    def _batch_size_from_signature(signature: _Signature) -> int:
        for key, value in signature:
            if key == "batch_size":
                return int(value)
        return 1
