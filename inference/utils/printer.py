import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from PIL import Image

def print_task_start(
    distributed: bool,
    rank: int,
    local_idx: int,
    assigned_total: int,
    global_position: int,
    total_tasks: int,
    task_id: str,
    task_kind: str,
) -> None:
    if distributed:
        print(
            f"[rank {rank}] [{local_idx}/{assigned_total}] "
            f"(global {global_position}/{total_tasks}) Running task '{task_id}' ({task_kind})"
        )
    else:
        print(f"[{global_position}/{total_tasks}] Running task '{task_id}' ({task_kind})")


def report_generation_time(generation_start: Optional[float], generation_end: Optional[float], distributed: bool, rank: int) -> None:
    if generation_start is None or generation_end is None:
        return
    elapsed = generation_end - generation_start
    message = f"Generation time (requests): {elapsed:.2f} seconds"
    if distributed:
        print(f"[rank {rank}] {message}")
    else:
        print(message)

# ******** Result aggregation and reporting helpers ********
def write_summary(
    summary_records: List[Tuple[int, Dict[str, Any]]],
    output_dir: Path,
    distributed: bool,
    rank: int,
    world_size: int,
) -> None:
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
        summary_records.sort(key=lambda item: item[0])
        summary = [entry for _, entry in summary_records]
        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"Completed {len(summary)} tasks. Summary written to {summary_path}")
