"""Public adaptive outcome tracking for CONVERA."""

from __future__ import annotations

import json
from pathlib import Path
import time

from config import LEARNING_HISTORY_PATH


class LearningLayer:
    def __init__(self, path: str | Path = LEARNING_HISTORY_PATH, *, window: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window = max(10, int(window))
        self.history: list[dict] = self._load()

    def record_outcome(self, features: dict, decision: dict, metrics: dict) -> None:
        entry = {
            "timestamp": time.time(),
            "features": _safe_subset(features, {"has_cache", "has_prefix", "similarity_bucket"}),
            "decision": _safe_subset(decision, {"mode", "precision"}),
            "metrics": _safe_subset(
                metrics,
                {
                    "latency_seconds",
                    "cache_hit",
                    "tokens_reused",
                    "tokens_computed",
                    "compute_avoided_pct",
                },
            ),
        }
        self.history.append(entry)
        self.history = self.history[-self.window :]
        self._save()

    def get_adjusted_thresholds(self) -> dict:
        reuse_runs = [
            item for item in self.history[-self.window :]
            if item.get("decision", {}).get("mode") in {"cache", "kv"}
        ]
        if not reuse_runs:
            return {"similarity": 0.85, "prefix_tokens": 8}

        successes = sum(1 for item in reuse_runs if item.get("metrics", {}).get("cache_hit"))
        success_rate = successes / len(reuse_runs)
        if success_rate >= 0.75:
            return {"similarity": 0.80, "prefix_tokens": 6}
        if success_rate <= 0.35:
            return {"similarity": 0.90, "prefix_tokens": 12}
        return {"similarity": 0.85, "prefix_tokens": 8}

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.history[-self.window :], indent=2), encoding="utf-8")


def _safe_subset(payload: dict, allowed: set[str]) -> dict:
    return {key: payload[key] for key in allowed if key in payload}
