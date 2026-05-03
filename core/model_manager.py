"""Multi-precision Hugging Face model manager for CONVERA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEFAULT_MODEL_PATH

log = logging.getLogger(__name__)

SUPPORTED_PRECISIONS = {"fp16", "int8", "int4"}


@dataclass(slots=True)
class ModelBundle:
    tokenizer: object
    model: object
    requested_precision: str
    actual_precision: str
    load_seconds: float


class ModelManager:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, *, device_map: str | None = "auto") -> None:
        self.model_path = Path(model_path)
        self.device_map = device_map
        self._models: dict[str, ModelBundle] = {}

    def load_model(self, precision: str = "fp16") -> ModelBundle:
        precision = _normalize_precision(precision)
        if precision in self._models:
            return self._models[precision]
        bundle = self._load(precision)
        self._models[precision] = bundle
        return bundle

    def get_model(self, precision: str = "fp16") -> ModelBundle:
        return self.load_model(precision)

    def loaded_precisions(self) -> list[str]:
        return sorted(self._models)

    def _load(self, precision: str) -> ModelBundle:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}. Download a model into models/llama3 first."
            )

        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        kwargs, actual_precision = _load_kwargs(precision, self.device_map)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        except Exception as exc:
            if precision == "fp16":
                raise
            log.warning("Falling back to fp16 load after %s failed: %s", precision, exc)
            kwargs, actual_precision = _load_kwargs("fp16", self.device_map)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        elapsed = time.time() - start
        print(f"[MODEL] Loaded {self.model_path} precision={actual_precision} in {elapsed:.2f}s")
        return ModelBundle(
            tokenizer=tokenizer,
            model=model,
            requested_precision=precision,
            actual_precision=actual_precision,
            load_seconds=elapsed,
        )


def _normalize_precision(precision: str) -> str:
    value = precision.lower().strip()
    if value not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision: {precision}. Expected one of {sorted(SUPPORTED_PRECISIONS)}")
    return value


def _load_kwargs(precision: str, device_map: str | None) -> tuple[dict, str]:
    kwargs: dict = {}
    if device_map:
        kwargs["device_map"] = device_map

    if precision == "fp16":
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        return kwargs, "fp16" if torch.cuda.is_available() else "fp32"

    if not torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float32
        return kwargs, "fp32"

    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        kwargs["torch_dtype"] = torch.float16
        return kwargs, "fp16"

    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        return kwargs, "int8"

    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return kwargs, "int4"
