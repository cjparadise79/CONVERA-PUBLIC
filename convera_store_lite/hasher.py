"""SHA-256 helpers for public content-addressed chunks."""

from __future__ import annotations

import hashlib

def hash_bytes(payload: bytes, *, algorithm: str = "sha256") -> str:
    del algorithm
    return hashlib.sha256(payload).hexdigest()
