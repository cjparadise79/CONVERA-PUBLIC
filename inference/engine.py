"""CONVERA inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid

import torch

from config import DEFAULT_AUDIT_MODE, DEFAULT_MAX_NEW_TOKENS, DEFAULT_VERIFICATION_MODE, INDEX_DIR
from convera_core_api import interface as core_api
from core.backend_selector import get_backend
from core.model_manager import ModelManager
from inference.cache_validator import CacheValidator
from inference.execution_inspector import ExecutionInspector
from inference.execution_router import ExecutionRouter
from inference.kv_manager import KVManager
from inference.learning_layer import LearningLayer
from inference.quantization_controller import QuantizationController
from inference.routing_model import RoutingModel
from audit.audit_logger import AuditLogger


@dataclass(slots=True)
class InferenceStats:
    request_id: str = ""
    latency_seconds: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    kv_hit: bool = False
    backend: str = "unknown"
    memory_mb: float = 0.0
    reused_prefill: bool = False
    fallback_reason: str | None = None
    precision: str = "fp16"
    requested_precision: str = "fp16"
    tokens_reused: int = 0
    tokens_computed: int = 0
    compute_avoided_pct: float = 0.0
    execution_mode: str = "full"
    route_verified: bool = False
    cache_validation: bool = False
    validation_latency_seconds: float = 0.0


@dataclass(slots=True)
class InferenceResult:
    text: str
    stats: InferenceStats = field(default_factory=InferenceStats)


class ConveraEngine:
    def __init__(
        self,
        model,
        tokenizer,
        kv_manager: KVManager | None = None,
        *,
        backend=None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        model_manager: ModelManager | None = None,
        quantization_controller: QuantizationController | None = None,
        precision_mode: str = "fp16",
        execution_router: ExecutionRouter | None = None,
        cache_validator: CacheValidator | None = None,
        learning_layer: LearningLayer | None = None,
        execution_inspector: ExecutionInspector | None = None,
        routing_model: RoutingModel | None = None,
        audit_logger: AuditLogger | None = None,
        audit_mode: bool = DEFAULT_AUDIT_MODE,
        verification_mode: bool = DEFAULT_VERIFICATION_MODE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.kv = kv_manager or KVManager()
        self.kv_by_precision: dict[str, KVManager] = {precision_mode if precision_mode != "auto" else "fp16": self.kv}
        self.backend = backend or get_backend()
        self.max_new_tokens = max_new_tokens
        self.model_manager = model_manager
        self.quantization_controller = quantization_controller or QuantizationController()
        self.precision_mode = precision_mode
        self.execution_router = execution_router or ExecutionRouter()
        self.cache_validator = cache_validator or CacheValidator()
        self.learning_layer = learning_layer or LearningLayer()
        self.execution_inspector = execution_inspector or ExecutionInspector()
        self.routing_model = routing_model or RoutingModel()
        self.audit_logger = audit_logger or AuditLogger()
        self.audit_mode = bool(audit_mode)
        self.verification_mode = bool(verification_mode)

    def run(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        precision_mode: str | None = None,
        audit_mode: bool | None = None,
    ) -> InferenceResult:
        run_audit = self.audit_mode if audit_mode is None else bool(audit_mode)
        max_tokens = int(max_new_tokens or self.max_new_tokens)
        tokenizer, model, kv_manager, requested_precision, actual_precision = self._runtime_for_request(
            precision_mode or self.precision_mode
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(self.backend, "device", "cpu")
        inputs = self.backend.move_inputs(inputs)

        cached = kv_manager.get(inputs.input_ids, device=device)
        prompt_len = int(inputs.input_ids.shape[-1])
        thresholds = self.learning_layer.get_adjusted_thresholds()
        route_features = {
            "exact_cache_hit": cached is not None,
            "prefix_match": False,
            "prompt_length": prompt_len,
            "previous_cache_success_rate": self.runtime_hit_rate(),
            "tokens_reused": 0,
        }
        learned_mode = self.routing_model.predict(route_features)
        decision = self.execution_router.route(
            prompt=prompt,
            cache_state={"exact_cache_hit": cached is not None},
            learned_mode=learned_mode,
            verification_mode=self.verification_mode,
        )
        stats = InferenceStats(
            request_id=str(uuid.uuid4()),
            kv_hit=cached is not None,
            backend=self.backend.name,
            precision=actual_precision,
            requested_precision=requested_precision,
            execution_mode=decision.mode,
            route_verified=decision.verified,
        )
        start = time.time()
        if run_audit:
            self.audit_logger.start_trace(stats.request_id, prompt_hash=self.cache_validator.hash_text(prompt))
            self.audit_logger.log_step(
                stats.request_id,
                "routing",
                {"mode": decision.mode, "source": "learned" if learned_mode else "deterministic"},
            )

        try:
            if cached is not None:
                cached = core_api.optimize_kv(cached)
            output, reused_prefill = self._generate(model, tokenizer, inputs, cached, max_tokens)
            stats.reused_prefill = reused_prefill
        except Exception as exc:
            stats.fallback_reason = str(exc)
            cached = None
            stats.kv_hit = False
            stats.execution_mode = "full"
            output, _ = self._generate(model, tokenizer, inputs, None, max_tokens)

        stats.latency_seconds = time.time() - start
        output_ids = _extract_sequences(output)
        stats.tokens_generated = max(0, int(output_ids.shape[-1]) - prompt_len)
        stats.tokens_reused = prompt_len if cached is not None else 0
        stats.tokens_computed = stats.tokens_generated + (0 if cached is not None else prompt_len)
        total_tokens = stats.tokens_reused + stats.tokens_computed
        stats.compute_avoided_pct = (
            (stats.tokens_reused / total_tokens) * 100.0 if total_tokens else 0.0
        )
        stats.tokens_per_second = (
            stats.tokens_generated / stats.latency_seconds if stats.latency_seconds > 0 else 0.0
        )
        stats.memory_mb = self.backend.get_memory_usage()
        if run_audit:
            self.audit_logger.log_step(
                stats.request_id,
                "execution",
                {
                    "kv_hit": stats.kv_hit,
                    "latency_seconds": stats.latency_seconds,
                    "tokens_reused": stats.tokens_reused,
                    "tokens_computed": stats.tokens_computed,
                },
            )

        self._store_prompt_kv(model, inputs, kv_manager)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if self.verification_mode or stats.kv_hit:
            validation_start = time.time()
            record = self.cache_validator.generate_record(prompt, decoded)
            stats.cache_validation = self.cache_validator.verify_record(record)
            stats.validation_latency_seconds = time.time() - validation_start
            self.cache_validator.store_record(record)
        if run_audit:
            self.audit_logger.log_step(
                stats.request_id,
                "validation",
                {"validated": stats.cache_validation, "latency_seconds": stats.validation_latency_seconds},
            )
        self._record_public_outcome(
            stats=stats,
            decision_mode=stats.execution_mode,
            requested_precision=requested_precision,
            route_features=route_features,
            thresholds=thresholds,
        )
        if run_audit:
            self.audit_logger.finalize_trace(
                stats.request_id,
                {
                    "precision": stats.precision,
                    "validated": stats.cache_validation,
                    "compute_avoided_pct": stats.compute_avoided_pct,
                },
            )
        return InferenceResult(text=decoded, stats=stats)

    def _record_public_outcome(
        self,
        *,
        stats: InferenceStats,
        decision_mode: str,
        requested_precision: str,
        route_features: dict,
        thresholds: dict,
    ) -> None:
        features = {
            "has_cache": stats.kv_hit,
            "has_prefix": stats.reused_prefill,
            "similarity_bucket": "public",
        }
        decision = {
            "mode": decision_mode,
            "precision": stats.precision,
        }
        metrics = {
            "latency_seconds": stats.latency_seconds,
            "cache_hit": stats.kv_hit,
            "tokens_reused": stats.tokens_reused,
            "tokens_computed": stats.tokens_computed,
            "compute_avoided_pct": stats.compute_avoided_pct,
        }
        self.learning_layer.record_outcome(features, decision, metrics)
        self.routing_model.record(route_features, decision, metrics)
        self.execution_inspector.log(
            {
                "request_id": stats.request_id,
                "mode": decision_mode,
                "precision": stats.precision,
                "requested_precision": requested_precision,
                "validated": stats.cache_validation,
                "thresholds": thresholds,
                "metrics": metrics,
                "steps": [
                    {
                        "name": "routing",
                        "decision": decision_mode,
                        "status": "deterministic",
                    },
                    {
                        "name": "precision",
                        "decision": stats.precision,
                        "status": "selected",
                    },
                    {
                        "name": "execution",
                        "decision": "cache_hit" if stats.kv_hit else "computed",
                        "status": "complete",
                    },
                    {
                        "name": "validation",
                        "decision": "passed" if stats.cache_validation else "not_run",
                        "status": "recorded" if stats.cache_validation else "available",
                    },
                ],
            }
        )

    def _generate(self, model, tokenizer, inputs, cached, max_new_tokens: int):
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if cached is None:
            with torch.no_grad():
                output = model.generate(**inputs, **generate_kwargs)
            return output, False

        # For exact prompt matches, reuse the cached prefill and continue from the final token.
        reuse_inputs = dict(inputs)
        reuse_inputs["input_ids"] = inputs.input_ids[:, -1:]
        if "attention_mask" in reuse_inputs:
            reuse_inputs["attention_mask"] = inputs.attention_mask
        with torch.no_grad():
            output = model.generate(
                **reuse_inputs,
                past_key_values=cached,
                **generate_kwargs,
            )
        return output, True

    def _store_prompt_kv(self, model, inputs, kv_manager: KVManager) -> None:
        with torch.no_grad():
            prefill = model(**inputs, use_cache=True)
        kv_manager.store(inputs.input_ids, getattr(prefill, "past_key_values", None))

    def _runtime_for_request(self, precision_mode: str):
        if self.model_manager is None:
            return self.tokenizer, self.model, self.kv, precision_mode, precision_mode

        if precision_mode == "auto":
            decision = self.quantization_controller.decide(reuse_score=self._reuse_score())
            requested_precision = decision.precision
        else:
            requested_precision = precision_mode

        bundle = self.model_manager.get_model(requested_precision)
        kv_manager = self._kv_for_precision(bundle.actual_precision)
        return bundle.tokenizer, bundle.model, kv_manager, requested_precision, bundle.actual_precision

    def _kv_for_precision(self, precision: str) -> KVManager:
        if precision not in self.kv_by_precision:
            self.kv_by_precision[precision] = KVManager(
                index_path=INDEX_DIR / f"kv_index_{precision}.json",
            )
        return self.kv_by_precision[precision]

    def _reuse_score(self) -> float:
        if not self.kv_by_precision:
            return 0.0
        return max(manager.hit_rate() for manager in self.kv_by_precision.values())

    def runtime_hit_rate(self) -> float:
        managers = _unique_managers([self.kv, *self.kv_by_precision.values()])
        hits = sum(manager.hits for manager in managers)
        misses = sum(manager.misses for manager in managers)
        total = hits + misses
        return hits / total if total else 0.0

    def runtime_chunk_reuse_ratio(self) -> float:
        managers = _unique_managers([self.kv, *self.kv_by_precision.values()])
        if not managers:
            return 0.0
        return max(manager.chunk_reuse_ratio() for manager in managers)


def _extract_sequences(output):
    if hasattr(output, "sequences"):
        return output.sequences
    return output


def _unique_managers(managers: list[KVManager]) -> list[KVManager]:
    seen: set[int] = set()
    unique: list[KVManager] = []
    for manager in managers:
        marker = id(manager)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(manager)
    return unique
