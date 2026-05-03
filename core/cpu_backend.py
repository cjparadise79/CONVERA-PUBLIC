"""CPU fallback backend."""

from __future__ import annotations

import os
import platform

from .backend import Backend, BackendInfo


class CPUBackend(Backend):
    name = "cpu"
    device = "cpu"

    def __init__(self, threads: int | None = None) -> None:
        self.threads = threads or max(1, (os.cpu_count() or 2) - 1)
        try:
            import torch

            torch.set_num_threads(self.threads)
        except Exception:
            pass

    def load_model(self, model):
        return model.to("cpu")

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            device=self.device,
            hardware=f"{platform.processor() or platform.machine()} ({self.threads} threads)",
            memory_mb=0.0,
        )

