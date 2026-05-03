"""Apple Metal Performance Shaders backend."""

from __future__ import annotations

from .backend import Backend, BackendInfo


class MPSBackend(Backend):
    name = "mps"
    device = "mps"

    def load_model(self, model):
        return model.to(self.device)

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, device=self.device, hardware="Apple MPS")

