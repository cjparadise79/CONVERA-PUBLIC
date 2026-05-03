"""Benchmark metrics helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path

from config import CHUNK_STORE_DIR


class Metrics:
    def __init__(self) -> None:
        self.start: float | None = None
        self.end: float | None = None

    def start_timer(self) -> None:
        self.start = time.time()

    def stop_timer(self) -> None:
        self.end = time.time()

    def latency(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start


def tokens_per_second(num_tokens: int, latency: float) -> float:
    return num_tokens / latency if latency > 0 else 0.0


def get_disk_usage(path: str | Path = CHUNK_STORE_DIR) -> int:
    root_path = Path(path)
    total = 0
    if not root_path.exists():
        return 0
    for root, _dirs, files in os.walk(root_path):
        for name in files:
            total += os.path.getsize(Path(root) / name)
    return total

