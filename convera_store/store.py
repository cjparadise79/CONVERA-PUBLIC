"""Tensor storage facade over the CONVERA storage boundary."""

from __future__ import annotations

from pathlib import Path

from config import CHUNK_STORE_DIR, MODEL_INDEX_PATH
from convera_core_api import interface as core_api
from convera_store_lite.index import JsonIndex
from convera_store_lite.store import ChunkStore


class ConveraTensorStore:
    def __init__(
        self,
        *,
        chunk_root: str | Path = CHUNK_STORE_DIR,
        index_path: str | Path = MODEL_INDEX_PATH,
        chunk_size: int | None = None,
    ) -> None:
        self.chunk_store = ChunkStore(chunk_root)
        self.index = JsonIndex(index_path)
        self.chunk_size = chunk_size

    def store_chunk(self, digest: str, tensor_block) -> bool:
        from convera_store_lite.chunker import tensor_to_bytes

        payload, _shape, _dtype = tensor_to_bytes(tensor_block)
        return self.chunk_store.store_chunk(digest, payload)

    def get_chunk(self, digest: str) -> bytes:
        return self.chunk_store.get_chunk(digest)

    def store_tensor(self, tensor, *, key: str | None = None) -> dict:
        if self.chunk_size is None:
            manifest = core_api.store_tensor(tensor)
        else:
            manifest = self.chunk_store.store_tensor(tensor, chunk_size=self.chunk_size)
        if key:
            self.index.store_mapping(key, manifest)
        return manifest

    def load_tensor(self, refs: dict | str, *, device: str | None = None):
        manifest = self.index.get_mapping(refs) if isinstance(refs, str) else refs
        if manifest is None:
            raise KeyError(f"No tensor mapping found for {refs!r}")
        tensor = core_api.load_tensor(manifest) if self.chunk_size is None else self.chunk_store.load_tensor(manifest)
        if device:
            tensor = tensor.to(device)
        return tensor

    def reconstruct_tensor(self, block_map: dict, *, device: str | None = None):
        return self.load_tensor(block_map, device=device)

    def reuse_ratio(self) -> float:
        return self.chunk_store.reuse_ratio()
