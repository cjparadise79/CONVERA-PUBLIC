from __future__ import annotations

import unittest

from inference.execution_router import ExecutionRouter
from inference.quantization_controller import QuantizationController


class ExecutionControlTests(unittest.TestCase):
    def test_execution_router_uses_public_modes(self) -> None:
        router = ExecutionRouter()

        self.assertEqual(router.route(exact_cache_hit=True).mode, "cache")
        self.assertEqual(router.route(cache_state={"exact_cache_hit": True}).mode, "cache")
        self.assertEqual(router.route(exact_cache_hit=False, prefix_match=True).mode, "kv")
        self.assertEqual(router.route(cache_state={"prefix_match": True}).mode, "kv")
        self.assertEqual(router.route(exact_cache_hit=False).mode, "full")

    def test_cache_validator_generates_public_record(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from inference.cache_validator import CacheValidator

        with TemporaryDirectory() as tmp:
            validator = CacheValidator(Path(tmp) / "validation_records.json")
            record = validator.generate_record("prompt", "response")

            self.assertEqual(record["method"], "deterministic-check")
            self.assertTrue(validator.verify_record(record))
            validator.store_record(record)

    def test_learning_layer_adjusts_public_thresholds(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from inference.learning_layer import LearningLayer

        with TemporaryDirectory() as tmp:
            layer = LearningLayer(Path(tmp) / "learning_history.json")
            self.assertEqual(layer.get_adjusted_thresholds()["similarity"], 0.85)
            for _ in range(4):
                layer.record_outcome(
                    {"has_cache": True},
                    {"mode": "kv", "precision": "fp16"},
                    {"cache_hit": True, "latency_seconds": 0.1, "tokens_reused": 4},
                )
            self.assertEqual(layer.get_adjusted_thresholds()["similarity"], 0.80)

    def test_execution_inspector_returns_latest_record(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from inference.execution_inspector import ExecutionInspector

        with TemporaryDirectory() as tmp:
            inspector = ExecutionInspector(Path(tmp) / "execution_records.json")
            inspector.log({"request_id": "abc", "mode": "full"})

            self.assertEqual(inspector.latest()["request_id"], "abc")
            self.assertEqual(inspector.get("abc")["mode"], "full")

    def test_routing_model_conservative_prediction(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from inference.routing_model import RoutingModel

        with TemporaryDirectory() as tmp:
            model = RoutingModel(Path(tmp) / "routing_model.json", min_samples=3)
            for _ in range(3):
                model.record(
                    {"exact_cache_hit": True, "prompt_length": 4},
                    {"mode": "cache"},
                    {"cache_hit": True, "compute_avoided_pct": 50.0, "latency_seconds": 0.1},
                )

            self.assertTrue(model.is_ready())
            self.assertEqual(model.predict({"exact_cache_hit": True}), "cache")
            self.assertIsNone(model.predict({"exact_cache_hit": False}))

    def test_audit_logger_and_exporter_redact_trace(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from audit.audit_logger import AuditLogger
        from audit.exporter import export_trace

        with TemporaryDirectory() as tmp:
            logger = AuditLogger(Path(tmp))
            logger.start_trace("abc", prompt_hash="hash")
            logger.log_step("abc", "routing", {"mode": "full", "prompt": "secret"})
            trace = logger.finalize_trace("abc", {"precision": "fp16"})

            dumped = export_trace(trace, format="json")
            self.assertIn('"mode": "full"', dumped)
            self.assertNotIn("secret", dumped)

    def test_quantization_controller_thresholds(self) -> None:
        controller = QuantizationController()

        self.assertEqual(controller.decide_precision(reuse_score=0.95), "int4")
        self.assertEqual(controller.decide_precision(reuse_score=0.75), "int8")
        self.assertEqual(controller.decide_precision(reuse_score=0.10), "fp16")


if __name__ == "__main__":
    unittest.main()
