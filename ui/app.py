"""FastAPI app for local CONVERA dashboard and prompt endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DEFAULT_AUDIT_MODE, DEFAULT_MODEL_PATH, DEFAULT_VERIFICATION_MODE, ensure_runtime_dirs
from ui.dashboard import dashboard

app = FastAPI(title="CONVERA")
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

_engine = None
BENCHMARK_GRAPH_PATH = Path("metrics/output/benchmark.png")


class Prompt(BaseModel):
    text: str
    max_new_tokens: int | None = None
    precision: str | None = None
    audit: bool | None = None


@app.get("/health")
def health():
    return {"ok": True, "engine_loaded": _engine is not None}


@app.get("/status")
def status():
    return {
        "ok": True,
        "engine_loaded": _engine is not None,
        "verification_mode": DEFAULT_VERIFICATION_MODE,
        "audit_mode": DEFAULT_AUDIT_MODE,
    }


@app.get("/metrics")
def get_metrics():
    return dashboard.get()


@app.get("/benchmark-graph")
def benchmark_graph():
    if not BENCHMARK_GRAPH_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark graph has not been generated yet")
    return FileResponse(BENCHMARK_GRAPH_PATH)


@app.get("/api/execution/latest")
def latest_execution():
    if _engine is None:
        raise HTTPException(status_code=404, detail="No execution records yet")
    engine = _engine
    record = engine.execution_inspector.latest()
    if record is None:
        raise HTTPException(status_code=404, detail="No execution records yet")
    return record


@app.get("/api/execution/{request_id}")
def execution_record(request_id: str):
    if _engine is None:
        raise HTTPException(status_code=404, detail="Execution record not found")
    engine = _engine
    record = engine.execution_inspector.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Execution record not found")
    return record


@app.get("/audit/{request_id}/export")
def audit_export(request_id: str, format: str = "json"):
    if _engine is None:
        raise HTTPException(status_code=404, detail="Audit trace not found")
    trace = _engine.audit_logger.load_trace(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Audit trace not found")
    from audit.exporter import export_trace

    try:
        payload = export_trace(trace, format=format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    media_type = "text/csv" if format == "csv" else "application/json"
    return PlainTextResponse(payload, media_type=media_type)


@app.get("/audit/{request_id}")
def audit_trace(request_id: str):
    if _engine is None:
        raise HTTPException(status_code=404, detail="Audit trace not found")
    trace = _engine.audit_logger.load_trace(request_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Audit trace not found")
    return trace


@app.post("/run")
def run_prompt(prompt: Prompt):
    engine = _get_engine()
    result = engine.run(
        prompt.text,
        max_new_tokens=prompt.max_new_tokens,
        precision_mode=prompt.precision,
        audit_mode=prompt.audit,
    )
    row = {
        "request_id": result.stats.request_id,
        "label": "UI RUN",
        "latency": result.stats.latency_seconds,
        "tps": result.stats.tokens_per_second,
        "disk": 0,
        "kv_hit": result.stats.kv_hit,
        "kv_hit_rate": engine.runtime_hit_rate(),
        "chunk_reuse": engine.runtime_chunk_reuse_ratio(),
        "tokens_generated": result.stats.tokens_generated,
        "tokens_computed": result.stats.tokens_computed,
        "tokens_reused": result.stats.tokens_reused,
        "compute_avoided_pct": result.stats.compute_avoided_pct,
        "cache_validation": result.stats.cache_validation,
        "validation_latency": result.stats.validation_latency_seconds,
        "backend": result.stats.backend,
        "memory_mb": result.stats.memory_mb,
        "precision": result.stats.precision,
        "requested_precision": result.stats.requested_precision,
        "execution_mode": result.stats.execution_mode,
    }
    dashboard.log(row)
    return {"output": result.text, "stats": row, "request_id": result.stats.request_id}


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    ensure_runtime_dirs()
    from core.model_manager import ModelManager
    from inference.engine import ConveraEngine
    from inference.kv_manager import KVManager

    manager = ModelManager(DEFAULT_MODEL_PATH)
    bundle = manager.get_model("fp16")
    _engine = ConveraEngine(
        bundle.model,
        bundle.tokenizer,
        KVManager(),
        model_manager=manager,
        precision_mode="auto",
    )
    return _engine
