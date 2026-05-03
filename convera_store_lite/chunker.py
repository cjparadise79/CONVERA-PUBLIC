"""Fixed-size byte and tensor chunking for CONVERA-OSS.

This module intentionally uses simple fixed-size chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from config import DEFAULT_CHUNK_SIZE
from .hasher import hash_bytes


@dataclass(slots=True)
class Chunk:
    digest: str
    offset: int
    size: int
    data: bytes


def chunk_bytes(payload: bytes, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Chunk]:
    size = max(1, int(chunk_size))
    chunks: list[Chunk] = []
    for offset in range(0, len(payload), size):
        data = payload[offset : offset + size]
        chunks.append(Chunk(digest=hash_bytes(data), offset=offset, size=len(data), data=data))
    if not payload:
        chunks.append(Chunk(digest=hash_bytes(b""), offset=0, size=0, data=b""))
    return chunks


def tensor_to_bytes(tensor) -> tuple[bytes, tuple[int, ...], str]:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(), tuple(int(dim) for dim in array.shape), str(array.dtype)


def bytes_to_tensor(payload: bytes, shape: Iterable[int], dtype: str):
    import torch

    array = np.frombuffer(payload, dtype=np.dtype(dtype)).copy().reshape(tuple(shape))
    return torch.from_numpy(array)


def chunk_tensor(tensor, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[list[Chunk], dict]:
    payload, shape, dtype = tensor_to_bytes(tensor)
    metadata = {
        "shape": list(shape),
        "dtype": dtype,
        "byte_length": len(payload),
        "sha256": hash_bytes(payload),
    }
    return chunk_bytes(payload, chunk_size=chunk_size), metadata
