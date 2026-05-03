"""Export public CONVERA audit traces."""

from __future__ import annotations

import csv
import io
import json


def export_trace(trace: dict, *, format: str = "json") -> str:
    if format == "json":
        return json.dumps(trace, indent=2)
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["request_id", "step", "timestamp", "data"])
        writer.writeheader()
        for step in trace.get("steps", []):
            writer.writerow(
                {
                    "request_id": trace.get("request_id", ""),
                    "step": step.get("name", ""),
                    "timestamp": step.get("timestamp", ""),
                    "data": json.dumps(step.get("data", {}), sort_keys=True),
                }
            )
        return output.getvalue()
    raise ValueError("format must be 'json' or 'csv'")
