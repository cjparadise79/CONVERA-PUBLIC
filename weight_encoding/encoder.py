"""Basic public weight encoding for model weights."""

from __future__ import annotations

from dataclasses import dataclass

from config import MODEL_INDEX_PATH
from convera_store import ConveraTensorStore


@dataclass(slots=True)
class WeightEncodingResult:
    total_tensors: int
    unique_payloads: int
    redundancy_ratio: float
    mapping: dict


class WeightEncoder:
    def __init__(self, store: ConveraTensorStore | None = None) -> None:
        self.store = store or ConveraTensorStore(index_path=MODEL_INDEX_PATH)
        self.map: dict[str, dict] = {}
        self.unique_hashes: set[str] = set()

    def encode(self, model) -> WeightEncodingResult:
        state = model.state_dict()
        for name, tensor in state.items():
            manifest = self.store.store_tensor(tensor, key=f"model/{name}")
            self.map[name] = manifest
            self.unique_hashes.add(manifest["sha256"])
        total = len(state)
        unique = len(self.unique_hashes)
        redundancy = 1.0 - (unique / total) if total else 0.0
        print(f"[CONVERA] Total tensors: {total}")
        print(f"[CONVERA] Unique tensor payloads: {unique}")
        print(f"[CONVERA] Redundancy ratio: {redundancy:.4f}")
        return WeightEncodingResult(total, unique, redundancy, self.map)
