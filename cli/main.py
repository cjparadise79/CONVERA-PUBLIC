"""Command line interface for CONVERA."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_MODEL_PATH, ensure_runtime_dirs


def main() -> None:
    ensure_runtime_dirs()
    parser = argparse.ArgumentParser(prog="convera")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run local inference")
    run.add_argument("--prompt", default=None)
    run.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    run.add_argument("--max-new-tokens", type=int, default=100)
    run.add_argument("--precision", choices=["auto", "fp16", "int8", "int4"], default="fp16")

    encode = sub.add_parser("encode", help="Encode model weights with the public CONVERA format")
    encode.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    sub.add_parser("benchmark", help="Run benchmark suite")
    sub.add_parser("health", help="Check local project health")

    args = parser.parse_args()
    if args.command == "health" or args.command is None:
        _health()
    elif args.command == "run":
        _run(args)
    elif args.command == "encode":
        _encode(args)
    elif args.command == "benchmark":
        from benchmarks.benchmark import main as benchmark_main

        benchmark_main([])


def _health() -> None:
    from config import CHUNK_STORE_DIR, DATA_DIR, MODELS_DIR

    print("[CONVERA] health ok")
    print(f"[PATH] models={MODELS_DIR}")
    print(f"[PATH] data={DATA_DIR}")
    print(f"[PATH] chunks={CHUNK_STORE_DIR}")


def _run(args) -> None:
    from core.model_manager import ModelManager
    from inference.engine import ConveraEngine
    from inference.kv_manager import KVManager

    manager = ModelManager(args.model_path)
    initial_precision = "fp16" if args.precision == "auto" else args.precision
    bundle = manager.get_model(initial_precision)
    engine = ConveraEngine(
        bundle.model,
        bundle.tokenizer,
        KVManager(),
        max_new_tokens=args.max_new_tokens,
        model_manager=manager,
        precision_mode=args.precision,
    )

    if args.prompt:
        result = engine.run(args.prompt)
        _print_result(result)
        return

    while True:
        prompt = input("CONVERA >> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            return
        if not prompt:
            continue
        result = engine.run(prompt)
        _print_result(result)


def _encode(args) -> None:
    from core.model_loader import load_model
    from weight_encoding.encoder import WeightEncoder

    _tokenizer, model = load_model(Path(args.model_path))
    result = WeightEncoder().encode(model)
    print(f"[CONVERA] encoded tensors={result.total_tensors} redundancy={result.redundancy_ratio:.4f}")


def _print_result(result) -> None:
    stats = result.stats
    print(result.text)
    print("\n[CONVERA]")
    print(f"- Latency: {stats.latency_seconds:.2f}s")
    print(f"- Tokens/sec: {stats.tokens_per_second:.2f}")
    print(f"- KV Cache: {'HIT' if stats.kv_hit else 'MISS'}")
    print(f"- Compute avoided: {stats.compute_avoided_pct:.1f}%")
    print(f"- Backend: {stats.backend}")
    print(f"- Precision: {stats.precision}")
    print(f"- Memory: {stats.memory_mb:.2f} MB")
    if stats.fallback_reason:
        print(f"- KV reuse fallback: {stats.fallback_reason}")
