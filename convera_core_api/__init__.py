"""Public black-box API boundary for optional CONVERA Core acceleration."""

from .interface import ADVANCED_MODE, load_tensor, merge_states, optimize_kv, store_tensor

__all__ = [
    "ADVANCED_MODE",
    "load_tensor",
    "merge_states",
    "optimize_kv",
    "store_tensor",
]

