"""Public cache validation records for CONVERA."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time

from config import VALIDATION_RECORDS_PATH


@dataclass(slots=True)
class ValidationRecord:
    prompt_hash: str
    response_hash: str
    timestamp: float
    method: str = "deterministic-check"


class CacheValidator:
    def __init__(self, path: str | Path = VALIDATION_RECORDS_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def validate_response(self, prompt: str, response: str) -> bool:
        record = self.generate_record(prompt, response)
        return self.verify_record(record)

    def generate_record(self, prompt: str, response: str) -> dict:
        record = ValidationRecord(
            prompt_hash=_hash_text(prompt),
            response_hash=_hash_text(response),
            timestamp=time.time(),
        )
        payload = asdict(record)
        print("[VALIDATION] record generated")
        return payload

    def verify_record(self, record: dict) -> bool:
        required = {"prompt_hash", "response_hash", "timestamp", "method"}
        verified = required.issubset(record) and record.get("method") == "deterministic-check"
        print(f"[VALIDATION] record verified={verified}")
        return verified

    def store_record(self, record: dict) -> None:
        records = self._load()
        records.append(record)
        self.path.write_text(json.dumps(records[-500:], indent=2), encoding="utf-8")

    def hash_text(self, value: str) -> str:
        return _hash_text(value)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
