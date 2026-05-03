"""Metrics-only telemetry payloads for CONVERA."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
import uuid


def build_payload(
    *,
    latency: float,
    tps: float,
    kv_hit_rate: float,
    chunk_reuse: float,
    disk_usage: int,
    gpu_name: str,
    vram_used: float,
    model_name: str,
    tokens_generated: int,
) -> dict:
    return {
        "report_kind": "convera-metrics-only",
        "privacy_mode": "no-file-names-or-paths",
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "customer_details": {},
        "file_count": 0,
        "bundle_count": 0,
        "failed_file_count": 0,
        "completed_with_warnings": False,
        "total_original_bytes": 0,
        "total_stored_bytes": int(disk_usage),
        "total_logical_encoded_bytes": int(disk_usage),
        "weighted_reduction_percent": round(float(chunk_reuse) * 100.0, 4),
        "all_hashes_match": True,
        "metrics": {
            "performance_metrics": {
                "latency_seconds": float(latency),
                "tokens_per_second": float(tps),
                "tokens_generated": int(tokens_generated),
            },
            "inference_metrics": {
                "kv_cache_hit_rate": float(kv_hit_rate),
                "chunk_reuse_ratio": float(chunk_reuse),
            },
            "gpu_metrics": {
                "gpu_name": gpu_name,
                "vram_used_mb": float(vram_used),
            },
            "model_metrics": {
                "model_name": model_name,
                "precision": "fp16",
                "quantized": False,
            },
            "environment_tags": {
                "engine": "convera",
                "mode": "inference",
                "deployment": "local",
                "system": platform.system(),
                "machine": platform.machine(),
            },
        },
        "failure_summary": {
            "total_failures": 0,
            "by_stage": [],
            "by_error_type": [],
            "by_stage_and_error_type": [],
        },
    }
