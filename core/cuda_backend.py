"""CUDA backend."""

from __future__ import annotations

from .backend import Backend, BackendInfo


class CUDABackend(Backend):
    name = "cuda"
    device = "cuda"

    def __init__(self) -> None:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True

    def load_model(self, model):
        return model.to(self.device)

    def get_memory_usage(self) -> float:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)

    def info(self) -> BackendInfo:
        import torch

        hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda-unavailable"
        return BackendInfo(
            name=self.name,
            device=self.device,
            hardware=hardware,
            memory_mb=self.get_memory_usage(),
        )

