"""Run real CONVERA benchmark passes."""

from __future__ import annotations

import argparse

from config import DEFAULT_MODEL_PATH, ensure_runtime_dirs

from benchmarks.report import generate_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--send-telemetry", action="store_true")
    args = parser.parse_args(argv)

    ensure_runtime_dirs()

    from core.model_loader import load_model
    from inference.engine import ConveraEngine
    from inference.kv_manager import KVManager
    from benchmarks.runner import run_test

    tokenizer, model = load_model(args.model_path)
    engine = ConveraEngine(model, tokenizer, KVManager())

    prompt1 = "Explain how neural networks work in detail."
    prompt2 = "Explain how neural networks work in detail with examples."

    results = [
        run_test(engine, prompt1, "COLD RUN", send_telemetry=args.send_telemetry),
        run_test(engine, prompt1, "WARM RUN", send_telemetry=args.send_telemetry),
        run_test(engine, prompt2, "VARIATION RUN", send_telemetry=args.send_telemetry),
    ]

    print("\n=== SUMMARY ===")
    cold = results[0]
    warm = results[1]
    if warm["latency"] > 0:
        print(f"Speedup (warm vs cold): {cold['latency'] / warm['latency']:.2f}x")
    print(f"Disk growth: {(results[-1]['disk'] - cold['disk']) / (1024 * 1024):.2f} MB")
    report = generate_report(results)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
