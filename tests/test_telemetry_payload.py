from __future__ import annotations

import json
import unittest

from telemetry.convera_payload import build_payload


class TelemetryPayloadTests(unittest.TestCase):
    def test_payload_matches_metrics_only_privacy_contract(self) -> None:
        payload = build_payload(
            latency=1.2,
            tps=55.0,
            kv_hit_rate=0.5,
            chunk_reuse=0.25,
            disk_usage=1024,
            gpu_name="cpu",
            vram_used=0,
            model_name="llama3",
            tokens_generated=20,
        )
        dumped = json.dumps(payload)

        self.assertEqual(payload["report_kind"], "convera-metrics-only")
        self.assertEqual(payload["privacy_mode"], "no-file-names-or-paths")
        self.assertNotIn("prompt", dumped.lower())
        self.assertNotIn("output", dumped.lower())
        self.assertNotIn("/Volumes", dumped)
        self.assertNotIn("files", payload)


if __name__ == "__main__":
    unittest.main()
