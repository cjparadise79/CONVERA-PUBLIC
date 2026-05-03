"""Non-blocking CONVERA telemetry sender."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_METRICS_API_URL = ""
METRICS_API_URL_ENV = "CONVERA_METRICS_API_URL"
METRICS_API_KEY_ENV = "CONVERA_METRICS_API_KEY"


class ConveraTelemetryClient:
    def __init__(self, *, url: str | None = None, api_key: str | None = None) -> None:
        self.url = (url or os.getenv(METRICS_API_URL_ENV) or DEFAULT_METRICS_API_URL).strip()
        self.api_key = (api_key or os.getenv(METRICS_API_KEY_ENV) or "").strip()

    def send(self, payload: dict) -> bool:
        if not self.url:
            print(f"[Telemetry] Skipped: {METRICS_API_URL_ENV} is not configured")
            return False
        if not self.api_key:
            print(f"[Telemetry] Skipped: {METRICS_API_KEY_ENV} is not configured")
            return False
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=3) as response:
                status = getattr(response, "status", response.getcode())
                body = response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"[Telemetry] Failed: {exc}")
            return False
        except Exception as exc:
            print(f"[Telemetry] Failed: {exc}")
            return False
        if not (200 <= int(status) < 300):
            print(f"[Telemetry] Server error: {status} {body}")
            return False
        print("[Telemetry] Sent successfully")
        return True
