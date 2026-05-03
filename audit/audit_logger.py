"""Public audit logging for CONVERA execution traces."""

from __future__ import annotations

import json
from pathlib import Path
import time

from config import AUDIT_LOG_DIR


class AuditLogger:
    def __init__(self, directory: str | Path = AUDIT_LOG_DIR) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, dict] = {}

    def start_trace(self, request_id: str, *, prompt_hash: str) -> dict:
        trace = {
            "request_id": request_id,
            "prompt_hash": prompt_hash,
            "started_at": time.time(),
            "steps": [],
            "status": "running",
        }
        self.active[request_id] = trace
        self._write(trace)
        return trace

    def log_step(self, request_id: str, step_name: str, data: dict) -> None:
        trace = self.active.get(request_id) or self.load_trace(request_id)
        if trace is None:
            return
        trace.setdefault("steps", []).append(
            {
                "name": step_name,
                "timestamp": time.time(),
                "data": _safe_data(data),
            }
        )
        self.active[request_id] = trace
        self._write(trace)

    def finalize_trace(self, request_id: str, summary: dict | None = None) -> dict | None:
        trace = self.active.pop(request_id, None) or self.load_trace(request_id)
        if trace is None:
            return None
        trace["finished_at"] = time.time()
        trace["status"] = "complete"
        trace["summary"] = _safe_data(summary or {})
        self._write(trace)
        return trace

    def load_trace(self, request_id: str) -> dict | None:
        path = self._path(request_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _write(self, trace: dict) -> None:
        self._path(str(trace["request_id"])).write_text(json.dumps(trace, indent=2), encoding="utf-8")

    def _path(self, request_id: str) -> Path:
        safe = "".join(ch for ch in request_id if ch.isalnum() or ch in {"-", "_"})
        return self.directory / f"{safe}.json"


def _safe_data(data: dict) -> dict:
    forbidden = {"prompt", "output", "text", "token_ids", "path", "authorization", "api_key"}
    return {key: value for key, value in data.items() if key.lower() not in forbidden}
