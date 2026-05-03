"""Hugging Face model loading with runtime metrics."""

from __future__ import annotations

import logging
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEFAULT_MODEL_PATH

log = logging.getLogger(__name__)


def load_model(
    path: str | Path = DEFAULT_MODEL_PATH,
    *,
    device_map: str | None = "auto",
    dtype: torch.dtype = torch.float16,
):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. Download a model into models/llama3 first."
        )

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"torch_dtype": dtype}
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.time() - start
    dtype_name = str(getattr(model, "dtype", dtype))
    memory_mb = _gpu_memory_mb()
    log.info("Loaded model from %s in %.2fs", model_path, load_time)
    print(f"[MODEL] Loaded {model_path} in {load_time:.2f}s")
    print(f"[MODEL] dtype={dtype_name}")
    print(f"[MODEL] VRAM={memory_mb:.2f} MB")
    return tokenizer, model


def _gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

