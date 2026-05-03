"""Public adaptive precision controller for CONVERA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PrecisionDecision:
    precision: str
    reuse_score: float
    similarity_score: float
    reason: str


class QuantizationController:
    def __init__(
        self,
        *,
        int4_threshold: float = 0.85,
        int8_threshold: float = 0.60,
        default_precision: str = "fp16",
    ) -> None:
        self.int4_threshold = float(int4_threshold)
        self.int8_threshold = float(int8_threshold)
        self.default_precision = default_precision

    def decide_precision(
        self,
        *,
        reuse_score: float,
        similarity_score: float = 0.0,
        latency_target_ms: int | None = None,
    ) -> str:
        return self.decide(
            reuse_score=reuse_score,
            similarity_score=similarity_score,
            latency_target_ms=latency_target_ms,
        ).precision

    def decide(
        self,
        *,
        reuse_score: float,
        similarity_score: float = 0.0,
        latency_target_ms: int | None = None,
    ) -> PrecisionDecision:
        score = max(0.0, min(1.0, max(float(reuse_score), float(similarity_score))))
        if score >= self.int4_threshold:
            precision = "int4"
            reason = "high_reuse"
        elif score >= self.int8_threshold:
            precision = "int8"
            reason = "balanced_reuse"
        else:
            precision = self.default_precision
            reason = "full_compute"

        if latency_target_ms is not None and latency_target_ms < 250 and precision == self.default_precision:
            precision = "int8"
            reason = "latency_target"

        print(f"[PRECISION] selected={precision} reuse_score={score:.3f} reason={reason}")
        return PrecisionDecision(
            precision=precision,
            reuse_score=float(reuse_score),
            similarity_score=float(similarity_score),
            reason=reason,
        )
