"""Public execution routing primitives for CONVERA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExecutionDecision:
    mode: str
    verified: bool = False
    reason: str = "computed"


class ExecutionRouter:
    def route(
        self,
        *,
        prompt: str = "",
        similarity_score: float = 0.0,
        cache_state: dict | None = None,
        exact_cache_hit: bool | None = None,
        prefix_match: bool | None = None,
        learned_mode: str | None = None,
        verification_mode: bool = False,
    ) -> ExecutionDecision:
        state = cache_state or {}
        exact = bool(state.get("exact_cache_hit", False) if exact_cache_hit is None else exact_cache_hit)
        prefix = bool(state.get("prefix_match", False) if prefix_match is None else prefix_match)
        if learned_mode in {"cache", "kv", "full"}:
            decision = ExecutionDecision(mode=learned_mode, reason="learned_public_pattern")
        elif exact:
            decision = ExecutionDecision(mode="cache", reason="exact_cache_match")
        elif prefix:
            decision = ExecutionDecision(mode="kv", reason="prefix_match")
        else:
            decision = ExecutionDecision(mode="full", reason="no_cache_match")

        if verification_mode:
            print(
                "[ROUTE] "
                f"mode={decision.mode} similarity={float(similarity_score):.3f} "
                f"verification={verification_mode}"
            )
        return decision
