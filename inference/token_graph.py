"""Token path graph for reusable inference states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for pure graph operations.
    torch = None

from config import TOKEN_GRAPH_PATH


def hash_state(tokens) -> str:
    if torch is not None and isinstance(tokens, torch.Tensor):
        payload = tokens.detach().cpu().contiguous().numpy().tobytes()
    else:
        payload = json.dumps([int(token) for token in tokens], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TokenGraph:
    def __init__(self, path: str | Path = TOKEN_GRAPH_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict[str, str]] = {}
        self._load()

    def add_node(self, state_hash: str, kv_ref: str | dict | None = None) -> None:
        self.nodes[state_hash] = {"kv_ref": kv_ref}

    def add_sequence(self, tokens, kv_ref: str | dict | None = None) -> str:
        flat = tokens.flatten() if torch is not None and isinstance(tokens, torch.Tensor) else list(tokens)
        final_hash = hash_state(flat)
        self.add_node(final_hash, kv_ref)
        for i in range(len(flat) - 1):
            state = flat[: i + 1]
            next_token = int(flat[i + 1])
            state_hash = hash_state(state)
            next_state_hash = hash_state(flat[: i + 2])
            self.edges.setdefault(state_hash, {})[str(next_token)] = next_state_hash
        self.save()
        return final_hash

    def find_longest_path(self, tokens) -> tuple[str | None, int]:
        flat = tokens.flatten() if torch is not None and isinstance(tokens, torch.Tensor) else list(tokens)
        for i in range(len(flat), 0, -1):
            candidate = hash_state(flat[:i])
            if candidate in self.nodes:
                return candidate, i
        return None, 0

    def save(self) -> None:
        tmp = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp.write_text(
            json.dumps({"nodes": self.nodes, "edges": self.edges}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.nodes = payload.get("nodes", {})
        self.edges = payload.get("edges", {})
