"""Minimal public benchmark for repeated-prompt cache behavior.

This runner intentionally reports only public-safe measurements: cache hit,
latency, and response length. It does not expose token traces, cache contents,
or runtime internals.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import DEFAULT_MODEL_PATH, ensure_runtime_dirs
from core.model_manager import ModelManager
from inference.engine import ConveraEngine
from inference.kv_manager import KVManager


TEST_PROMPTS = [
    "Explain quantum mechanics simply",
    "Explain quantum mechanics simply",
    "Explain quantum mechanics in simple terms",
    "Explain quantum mechanics simply",
]
OUTPUT_DIR = "metrics/output"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "benchmark.json")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "benchmark.png")
COMPARISON_PATH = os.path.join(OUTPUT_DIR, "comparison.png")


def run_benchmark(*, model_path: str, max_new_tokens: int, precision: str) -> list[dict]:
    ensure_runtime_dirs()
    manager = ModelManager(model_path)
    initial_precision = "fp16" if precision == "auto" else precision
    bundle = manager.get_model(initial_precision)
    engine = ConveraEngine(
        bundle.model,
        bundle.tokenizer,
        KVManager(),
        max_new_tokens=max_new_tokens,
        model_manager=manager,
        precision_mode=precision,
    )

    results: list[dict] = []
    print("\n=== CONVERA BENCHMARK ===\n")
    for index, prompt in enumerate(TEST_PROMPTS, start=1):
        start = time.time()
        result = engine.run(prompt, max_new_tokens=max_new_tokens)
        latency_ms = int((time.time() - start) * 1000)
        entry = {
            "run": index,
            "request_id": result.stats.request_id,
            "cached": bool(result.stats.kv_hit),
            "latency": latency_ms,
            "response_len": len(result.text),
            "tokens_computed": result.stats.tokens_computed,
            "tokens_reused": result.stats.tokens_reused,
            "compute_avoided_pct": round(result.stats.compute_avoided_pct, 2),
            "precision": result.stats.precision,
            "execution_mode": result.stats.execution_mode,
            "validated": bool(result.stats.cache_validation),
            "validation_latency": result.stats.validation_latency_seconds,
        }
        results.append(entry)
        print(f"Run {index}:")
        print(f"  Cached: {entry['cached']}")
        print(f"  Latency: {entry['latency']} ms")
        print(f"  Compute avoided: {entry['compute_avoided_pct']}%")
        print()
    return results


def summarize(results: list[dict]) -> None:
    if not results:
        print("No results recorded")
        return

    total = len(results)
    cached_runs = sum(1 for result in results if result["cached"])
    avg_latency = sum(result["latency"] for result in results) / total
    avg_avoided = sum(result["compute_avoided_pct"] for result in results) / total
    validated_runs = sum(1 for result in results if result.get("validated"))
    validation_latency = sum(result.get("validation_latency", 0.0) for result in results)
    first_latency = results[0]["latency"]
    subsequent = results[1:]
    later_latency = sum(result["latency"] for result in subsequent) / len(subsequent) if subsequent else 0

    print("=== SUMMARY ===")
    print(f"Total runs: {total}")
    print(f"Cache hits: {cached_runs}")
    print(f"Avg latency: {int(avg_latency)} ms")
    print(f"Avg compute avoided: {avg_avoided:.1f}%")
    print(f"Validated runs: {validated_runs}")
    print(f"Validation overhead: {validation_latency:.4f}s")
    print(f"First run latency: {first_latency} ms")
    print(f"Subsequent avg latency: {int(later_latency)} ms")
    if subsequent and later_latency < first_latency:
        improvement = ((first_latency - later_latency) / first_latency) * 100
        print(f"Improvement: {int(improvement)}% faster on repeat")
    else:
        print("No improvement detected")


def save_results(results: list[dict]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    return RESULTS_PATH


def generate_graph(results: list[dict]) -> str | None:
    if not results:
        return None

    runs = [result["run"] for result in results]
    latencies = [result["latency"] for result in results]
    cached = [result["cached"] for result in results]
    colors = ["#1f8f4d" if value else "#b33a3a" for value in cached]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(runs, latencies, color="#243447", linewidth=1.8)
    plt.scatter(runs, latencies, c=colors, s=72, zorder=3)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Cached", markerfacecolor="#1f8f4d", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Computed", markerfacecolor="#b33a3a", markersize=8),
    ]
    plt.legend(handles=legend_elements)
    plt.title("CONVERA Benchmark")
    plt.xlabel("Run")
    plt.ylabel("Latency (ms)")
    plt.xticks(runs)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(GRAPH_PATH, dpi=160)
    plt.close()
    print(f"Graph saved to: {GRAPH_PATH}")
    return GRAPH_PATH


def generate_comparison_graph(results: list[dict]) -> str | None:
    cached_latencies = [result["latency"] for result in results if result["cached"]]
    computed_latencies = [result["latency"] for result in results if not result["cached"]]
    if not cached_latencies or not computed_latencies:
        return None

    cached_avg = sum(cached_latencies) / len(cached_latencies)
    computed_avg = sum(computed_latencies) / len(computed_latencies)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["Computed", "Cached"], [computed_avg, cached_avg], color=["#b33a3a", "#1f8f4d"])
    plt.title("Average Latency")
    plt.ylabel("Latency (ms)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(COMPARISON_PATH, dpi=160)
    plt.close()
    print(f"Comparison graph saved to: {COMPARISON_PATH}")
    return COMPARISON_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public CONVERA repeated-prompt benchmark.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--precision", choices=["auto", "fp16", "int8", "int4"], default="fp16")
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
    )
    summarize(results)
    save_results(results)
    generate_graph(results)
    generate_comparison_graph(results)


if __name__ == "__main__":
    main()
