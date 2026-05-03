"""Benchmark runner."""

from __future__ import annotations

from benchmarks.metrics import Metrics, get_disk_usage, tokens_per_second
from telemetry.convera_client import ConveraTelemetryClient
from telemetry.convera_payload import build_payload
from ui.dashboard import dashboard


def run_test(engine, prompt: str, label: str, *, send_telemetry: bool = False) -> dict:
    metrics = Metrics()
    print(f"\n--- {label} ---")
    print(f"Prompt: {prompt}\n")

    metrics.start_timer()
    result = engine.run(prompt)
    metrics.stop_timer()

    latency = metrics.latency()
    token_count = result.stats.tokens_generated or len(result.text.split())
    tps = result.stats.tokens_per_second or tokens_per_second(token_count, latency)
    disk = get_disk_usage()
    kv_hit_rate = engine.kv.hit_rate()
    chunk_reuse = engine.kv.chunk_reuse_ratio()

    print(f"Latency: {latency:.2f}s")
    print(f"Tokens/sec: {tps:.2f}")
    print(f"KV hit: {result.stats.kv_hit}")
    print(f"KV hit rate: {kv_hit_rate:.2f}")
    print(f"Chunk reuse: {chunk_reuse:.2f}")
    print(f"Disk usage: {disk / (1024 * 1024):.2f} MB")

    row = {
        "label": label,
        "latency": latency,
        "tps": tps,
        "disk": disk,
        "kv_hit": result.stats.kv_hit,
        "kv_hit_rate": kv_hit_rate,
        "chunk_reuse": chunk_reuse,
        "tokens_generated": token_count,
        "backend": result.stats.backend,
        "memory_mb": result.stats.memory_mb,
    }
    dashboard.log(row)

    if send_telemetry:
        payload = build_payload(
            latency=latency,
            tps=tps,
            kv_hit_rate=kv_hit_rate,
            chunk_reuse=chunk_reuse,
            disk_usage=disk,
            gpu_name=result.stats.backend,
            vram_used=result.stats.memory_mb,
            model_name="llama3",
            tokens_generated=token_count,
        )
        ConveraTelemetryClient().send(payload)

    return row

