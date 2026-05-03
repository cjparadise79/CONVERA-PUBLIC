"""Strict CONVERA API boundary.

This module exposes a data contract only. The public fallback is intentionally
simple and returns tensors, references, and minimal metadata.
"""

from __future__ import annotations

from config import CHUNK_STORE_DIR
from convera_store_lite.store import ChunkStore

__all__ = [
    "ADVANCED_MODE",
    "load_tensor",
    "merge_states",
    "optimize_kv",
    "store_tensor",
]

ADVANCED_MODE = False
_LITE_STORE = ChunkStore(CHUNK_STORE_DIR)


def store_tensor(tensor) -> dict:
    return _LITE_STORE.store_tensor(tensor)


def load_tensor(refs: dict):
    return _LITE_STORE.load_tensor(refs)


def optimize_kv(kv_tensor):
    return kv_tensor


def merge_states(state_a, state_b):
    return state_b if state_b is not None else state_a
