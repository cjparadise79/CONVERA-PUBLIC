"""Small JSON index for tensor and KV manifests."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any


class JsonIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.index: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(f"{self.path.suffix}.tmp")
            tmp.write_text(json.dumps(self.index, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)

    def store_mapping(self, key: str, refs: Any) -> None:
        with self._lock:
            self.index[key] = refs
            self.save()

    def get_mapping(self, key: str) -> Any:
        return self.index.get(key)

    def keys(self):
        return self.index.keys()

