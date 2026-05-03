"""Persistent KV cache manager backed by the CONVERA tensor store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from config import DEFAULT_KV_CHUNK_SIZE, KV_INDEX_PATH
from convera_store import ConveraTensorStore
from convera_store_lite.index import JsonIndex


class KVManager:
    def __init__(
        self,
        *,
        index_path: str | Path = KV_INDEX_PATH,
        tensor_store: ConveraTensorStore | None = None,
        gpu_cache_limit: int = 4,
    ) -> None:
        index_file = Path(index_path)
        self.index = JsonIndex(index_path)
        self.tensor_store = tensor_store or ConveraTensorStore(
            index_path=index_file.with_name("kv_tensor_index.json"),
            chunk_size=DEFAULT_KV_CHUNK_SIZE,
        )
        self.gpu_cache: dict[str, Any] = {}
        self.gpu_cache_order: list[str] = []
        self.hits = 0
        self.misses = 0
        self.gpu_cache_limit = max(0, int(gpu_cache_limit))

    def hash_tokens(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            payload = tokens.detach().cpu().contiguous().numpy().tobytes()
        else:
            payload = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, tokens, *, device: str | None = None):
        key = self.hash_tokens(tokens)
        cached = self.gpu_cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        manifest = self.index.get_mapping(key)
        if manifest is None:
            self.misses += 1
            return None

        kv = self._deserialize_kv(manifest, device=device)
        self.hits += 1
        self._remember_gpu(key, kv)
        return kv

    def store(self, tokens, kv) -> str | None:
        if kv is None:
            return None
        key = self.hash_tokens(tokens)
        serializable = self._serialize_kv(kv, key)
        self.index.store_mapping(key, serializable)
        self._remember_gpu(key, kv)
        return key

    def has(self, tokens) -> bool:
        return self.hash_tokens(tokens) in self.index.index

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def chunk_reuse_ratio(self) -> float:
        return self.tensor_store.reuse_ratio()

    def _serialize_kv(self, kv, key: str) -> dict:
        legacy = _to_legacy_cache(kv)
        layers = []
        for layer_idx, layer in enumerate(legacy):
            tensors = []
            for tensor_idx, tensor in enumerate(layer):
                if tensor is None:
                    tensors.append({"none": True})
                    continue
                manifest = self.tensor_store.store_tensor(
                    tensor,
                    key=f"kv/{key}/layer-{layer_idx}/tensor-{tensor_idx}",
                )
                tensors.append(manifest)
            layers.append(tensors)
        return {"format": "legacy_tuple_v1", "layers": layers}

    def _deserialize_kv(self, manifest: dict, *, device: str | None = None):
        layers = []
        for layer in manifest.get("layers", []):
            tensors = [
                None if tensor_manifest.get("none") else self.tensor_store.load_tensor(tensor_manifest, device=device)
                for tensor_manifest in layer
            ]
            layers.append(tuple(tensors))
        return tuple(layers)

    def _remember_gpu(self, key: str, kv) -> None:
        if self.gpu_cache_limit <= 0:
            return
        self.gpu_cache[key] = kv
        if key in self.gpu_cache_order:
            self.gpu_cache_order.remove(key)
        self.gpu_cache_order.append(key)
        while len(self.gpu_cache_order) > self.gpu_cache_limit:
            old_key = self.gpu_cache_order.pop(0)
            self.gpu_cache.pop(old_key, None)


def _to_legacy_cache(kv):
    if hasattr(kv, "to_legacy_cache"):
        return kv.to_legacy_cache()
    return kv
