"""In-process dashboard state for benchmark runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from config import METRICS_HISTORY_PATH


class DashboardState:
    def __init__(self, path: str | Path = METRICS_HISTORY_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = self._load()

    def log(self, entry: dict) -> None:
        payload = dict(entry)
        payload.setdefault("timestamp", time.time())
        self.history.append(payload)
        self.save()

    def get(self) -> list[dict]:
        return self.history

    def save(self) -> None:
        self.path.write_text(json.dumps(self.history[-500:], indent=2), encoding="utf-8")

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []


dashboard = DashboardState()

