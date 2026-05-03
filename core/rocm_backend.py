"""ROCm backend.

PyTorch exposes ROCm devices through the CUDA API, so this backend intentionally uses
`cuda` as the device while labeling the hardware path separately.
"""

from __future__ import annotations

from .cuda_backend import CUDABackend


class ROCmBackend(CUDABackend):
    name = "rocm"

    @staticmethod
    def is_rocm_runtime() -> bool:
        try:
            import torch

            return bool(getattr(torch.version, "hip", None))
        except Exception:
            return False

