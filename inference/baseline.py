"""Baseline inference entry point."""

from __future__ import annotations

from core.model_loader import load_model
from inference.engine import ConveraEngine
from inference.kv_manager import KVManager


def run_inference(prompt: str, *, model_path: str = "./models/llama3") -> str:
    tokenizer, model = load_model(model_path)
    engine = ConveraEngine(model, tokenizer, KVManager())
    result = engine.run(prompt)
    print(f"[TIME] {result.stats.latency_seconds:.2f}s")
    print(f"[TOKENS/SEC] {result.stats.tokens_per_second:.2f}")
    print(f"[KV] {'HIT' if result.stats.kv_hit else 'MISS'}")
    print(f"[GPU MB] {result.stats.memory_mb:.2f}")
    return result.text

