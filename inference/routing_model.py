"""Lightweight public routing model for CONVERA."""

from __future__ import annotations

import json
from pathlib import Path
import time

from config import ROUTING_MODEL_PATH


class RoutingModel:
    def __init__(self, path: str | Path = ROUTING_MODEL_PATH, *, min_samples: int = 8) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_samples = max(3, int(min_samples))
        self.samples: list[dict] = []
        self.mode_scores: dict[str, float] = {}
        self.load_model()

    def train(self, data: list[dict]) -> None:
        self.samples = list(data)[-500:]
        self._fit()
        self.save_model()

    def record(self, features: dict, decision: dict, metrics: dict) -> None:
        self.samples.append(
            {
                "timestamp": time.time(),
                "features": _public_features(features),
                "decision": {"mode": str(decision.get("mode", "full"))},
                "metrics": _public_metrics(metrics),
            }
        )
        self.samples = self.samples[-500:]
        if len(self.samples) % self.min_samples == 0:
            self._fit()
        self.save_model()

    def predict(self, features: dict) -> str | None:
        if not self.is_ready():
            return None
        exact = bool(features.get("exact_cache_hit"))
        prefix = bool(features.get("prefix_match"))
        if exact and self.mode_scores.get("cache", 0.0) >= 0.45:
            return "cache"
        if prefix and self.mode_scores.get("kv", 0.0) >= 0.45:
            return "kv"
        return None

    def is_ready(self) -> bool:
        return len(self.samples) >= self.min_samples and bool(self.mode_scores)

    def save_model(self) -> None:
        payload = {
            "samples": self.samples[-500:],
            "mode_scores": self.mode_scores,
            "min_samples": self.min_samples,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_model(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.samples = payload.get("samples", []) if isinstance(payload.get("samples"), list) else []
        scores = payload.get("mode_scores", {})
        self.mode_scores = scores if isinstance(scores, dict) else {}

    def _fit(self) -> None:
        grouped: dict[str, list[float]] = {}
        for sample in self.samples:
            mode = sample.get("decision", {}).get("mode", "full")
            metrics = sample.get("metrics", {})
            score = _score(metrics)
            grouped.setdefault(mode, []).append(score)
        self.mode_scores = {
            mode: sum(values) / len(values)
            for mode, values in grouped.items()
            if values
        }


def _score(metrics: dict) -> float:
    hit_bonus = 1.0 if metrics.get("cache_hit") else 0.0
    avoided = float(metrics.get("compute_avoided_pct", 0.0)) / 100.0
    latency = float(metrics.get("latency_seconds", 0.0))
    latency_bonus = 1.0 / (1.0 + max(0.0, latency))
    return (hit_bonus * 0.5) + (avoided * 0.35) + (latency_bonus * 0.15)


def _public_features(features: dict) -> dict:
    allowed = {"exact_cache_hit", "prefix_match", "prompt_length", "previous_cache_success_rate", "tokens_reused"}
    return {key: features[key] for key in allowed if key in features}


def _public_metrics(metrics: dict) -> dict:
    allowed = {"latency_seconds", "cache_hit", "tokens_reused", "tokens_computed", "compute_avoided_pct"}
    return {key: metrics[key] for key in allowed if key in metrics}
