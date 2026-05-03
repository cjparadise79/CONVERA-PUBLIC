"""Public execution inspection records for CONVERA."""

from __future__ import annotations

import json
from pathlib import Path
import time

from config import EXECUTION_RECORDS_PATH


class ExecutionInspector:
    def __init__(self, path: str | Path = EXECUTION_RECORDS_PATH, *, limit: int = 500) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = max(20, int(limit))
        self.records: list[dict] = self._load()

    def log(self, record: dict) -> dict:
        payload = dict(record)
        payload.setdefault("timestamp", time.time())
        self.records.append(payload)
        self.records = self.records[-self.limit :]
        self._save()
        return payload

    def get(self, request_id: str) -> dict | None:
        for record in reversed(self.records):
            if record.get("request_id") == request_id:
                return record
        return None

    def latest(self) -> dict | None:
        return self.records[-1] if self.records else None

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.records[-self.limit :], indent=2), encoding="utf-8")
