"""Writable lite chunk store for CONVERA-OSS tensors.

This store demonstrates persistence and deduplication with fixed-size chunks.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
import uuid
import zlib

from config import CHUNK_STORE_DIR, DEFAULT_CHUNK_SIZE
from .chunker import bytes_to_tensor, chunk_bytes, chunk_tensor
from .hasher import hash_bytes


@dataclass(slots=True)
class StoreStats:
    total_chunks: int = 0
    new_chunks: int = 0
    reused_chunks: int = 0
    logical_bytes: int = 0
    stored_bytes: int = 0

    @property
    def reuse_ratio(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.reused_chunks / self.total_chunks


class ChunkStore:
    def __init__(
        self,
        root: str | Path = CHUNK_STORE_DIR,
        *,
        compression_level: int = 6,
        ram_cache_items: int = 512,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.ram_cache_items = max(0, int(ram_cache_items))
        self.ram_cache: OrderedDict[str, bytes] = OrderedDict()
        self.stats = StoreStats()

    def chunk_path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / f"{digest}.chunk"

    def store_chunk(self, digest: str, payload: bytes) -> bool:
        self.stats.total_chunks += 1
        self.stats.logical_bytes += len(payload)
        self._remember(digest, payload)

        chunk_path = self.chunk_path(digest)
        if chunk_path.exists():
            self.stats.reused_chunks += 1
            return False

        encoded = zlib.compress(payload, self.compression_level)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = chunk_path.with_name(f".{chunk_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, chunk_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self.stats.new_chunks += 1
        self.stats.stored_bytes += len(encoded)
        return True

    def get_chunk(self, digest: str) -> bytes:
        cached = self.ram_cache.get(digest)
        if cached is not None:
            self.ram_cache.move_to_end(digest)
            return cached

        chunk_path = self.chunk_path(digest)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing CONVERA chunk: {digest}")
        payload = zlib.decompress(chunk_path.read_bytes())
        if hash_bytes(payload) != digest:
            raise ValueError(f"Chunk hash mismatch for {digest}")
        self._remember(digest, payload)
        return payload

    def store_bytes(self, payload: bytes, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict]:
        refs: list[dict] = []
        for chunk in chunk_bytes(payload, chunk_size=chunk_size):
            self.store_chunk(chunk.digest, chunk.data)
            refs.append({"hash": chunk.digest, "offset": chunk.offset, "size": chunk.size})
        return refs

    def load_bytes(self, refs: list[dict], *, expected_sha256: str | None = None) -> bytes:
        payload = b"".join(self.get_chunk(ref["hash"]) for ref in refs)
        if expected_sha256 and hash_bytes(payload) != expected_sha256:
            raise ValueError("Reconstructed payload hash mismatch")
        return payload

    def store_tensor(self, tensor, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
        chunks, metadata = chunk_tensor(tensor, chunk_size=chunk_size)
        refs = []
        for chunk in chunks:
            self.store_chunk(chunk.digest, chunk.data)
            refs.append({"hash": chunk.digest, "offset": chunk.offset, "size": chunk.size})
        metadata["chunks"] = refs
        metadata["chunk_size"] = int(chunk_size)
        return metadata

    def load_tensor(self, manifest: dict, *, device: str | None = None):
        payload = self.load_bytes(manifest["chunks"], expected_sha256=manifest.get("sha256"))
        tensor = bytes_to_tensor(payload, manifest["shape"], manifest["dtype"])
        if device:
            tensor = tensor.to(device)
        return tensor

    def reuse_ratio(self) -> float:
        return self.stats.reuse_ratio

    def _remember(self, digest: str, payload: bytes) -> None:
        if self.ram_cache_items <= 0:
            return
        self.ram_cache[digest] = payload
        self.ram_cache.move_to_end(digest)
        while len(self.ram_cache) > self.ram_cache_items:
            self.ram_cache.popitem(last=False)
