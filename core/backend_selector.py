"""Hardware backend detection."""

from __future__ import annotations

from .cpu_backend import CPUBackend
from .cuda_backend import CUDABackend
from .mps_backend import MPSBackend
from .rocm_backend import ROCmBackend


def get_backend(prefer: str | None = None):
    import torch

    requested = (prefer or "").lower().strip()
    if requested == "cpu":
        return CPUBackend()
    if requested == "mps":
        return MPSBackend()
    if requested in {"cuda", "gpu"}:
        return CUDABackend()

    if torch.cuda.is_available():
        if ROCmBackend.is_rocm_runtime():
            return ROCmBackend()
        return CUDABackend()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return MPSBackend()
    return CPUBackend()

