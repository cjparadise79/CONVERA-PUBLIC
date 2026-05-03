"""Execution backend abstractions."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(slots=True)
class BackendInfo:
    name: str
    device: str
    hardware: str
    memory_mb: float = 0.0


class Backend:
    name = "base"
    device = "cpu"

    def load_model(self, model: Any):
        return model

    def move_inputs(self, inputs: Any):
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        return inputs

    def run_inference(self, model: Any, inputs: dict, **generate_kwargs):
        import torch

        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)
        return output, time.time() - start

    def get_memory_usage(self) -> float:
        return 0.0

    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self.name,
            device=self.device,
            hardware=self.name,
            memory_mb=self.get_memory_usage(),
        )

