"""Reconstruct model state dictionaries from public weight manifests."""

from __future__ import annotations

import torch

from convera_store import ConveraTensorStore


class WeightReconstructor:
    def __init__(self, store: ConveraTensorStore | None = None) -> None:
        self.store = store or ConveraTensorStore()

    def reconstruct_state_dict(self, mapping: dict[str, dict], *, device: str | None = None) -> dict:
        return {name: self.store.load_tensor(manifest, device=device) for name, manifest in mapping.items()}

    def load_into_model(self, model, mapping: dict[str, dict], *, strict: bool = True):
        state = self.reconstruct_state_dict(mapping)
        return model.load_state_dict(state, strict=strict)

    def validate(self, original_state: dict, reconstructed_state: dict) -> bool:
        for name, original in original_state.items():
            candidate = reconstructed_state[name]
            if not torch.equal(original.detach().cpu(), candidate.detach().cpu()):
                return False
        return True
