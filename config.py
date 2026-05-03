"""Project paths and runtime defaults for CONVERA."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "llama3"

DATA_DIR = PROJECT_ROOT / "data"
CHUNK_STORE_DIR = DATA_DIR / "chunks"
INDEX_DIR = DATA_DIR / "index"
KV_INDEX_PATH = INDEX_DIR / "kv_index.json"
MODEL_INDEX_PATH = INDEX_DIR / "model_index.json"
TOKEN_GRAPH_PATH = INDEX_DIR / "token_graph.json"
METRICS_HISTORY_PATH = DATA_DIR / "metrics_history.json"
VALIDATION_RECORDS_PATH = INDEX_DIR / "validation_records.json"
LEARNING_HISTORY_PATH = DATA_DIR / "learning_history.json"
EXECUTION_RECORDS_PATH = DATA_DIR / "execution_records.json"
ROUTING_MODEL_PATH = DATA_DIR / "routing_model.json"
AUDIT_LOG_DIR = DATA_DIR / "audit_logs"

DEFAULT_CHUNK_SIZE = int(os.getenv("CONVERA_CHUNK_SIZE", str(4 * 1024 * 1024)))
DEFAULT_KV_CHUNK_SIZE = int(os.getenv("CONVERA_KV_CHUNK_SIZE", str(512 * 1024)))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("CONVERA_MAX_NEW_TOKENS", "100"))
DEFAULT_TELEMETRY_ENABLED = os.getenv("CONVERA_TELEMETRY", "0") == "1"
DEFAULT_VERIFICATION_MODE = os.getenv("CONVERA_VERIFICATION_MODE", "0") == "1"
DEFAULT_AUDIT_MODE = os.getenv("CONVERA_AUDIT_MODE", "0") == "1"


def ensure_runtime_dirs() -> None:
    for path in (MODELS_DIR, DATA_DIR, CHUNK_STORE_DIR, INDEX_DIR, AUDIT_LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
