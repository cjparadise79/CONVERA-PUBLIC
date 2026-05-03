"""Generate a compact benchmark report."""

from __future__ import annotations

import json
from pathlib import Path


def generate_report(results: list[dict], filename: str = "convera_report.pdf") -> Path:
    target = Path(filename)
    try:
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table
    except Exception:
        fallback = target.with_suffix(".json")
        fallback.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return fallback

    doc = SimpleDocTemplate(str(target))
    styles = getSampleStyleSheet()
    rows = [["Run", "Latency", "Tokens/sec", "KV Hit", "Chunk Reuse", "Disk MB"]]
    for row in results:
        rows.append(
            [
                row["label"],
                f"{row['latency']:.2f}s",
                f"{row['tps']:.2f}",
                "yes" if row.get("kv_hit") else "no",
                f"{row.get('chunk_reuse', 0.0):.2f}",
                f"{row['disk'] / (1024 * 1024):.2f}",
            ]
        )

    content = [
        Paragraph("CONVERA Benchmark Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Metrics-only report. Prompt text and model output are intentionally omitted.", styles["BodyText"]),
        Spacer(1, 12),
        Table(rows),
    ]
    doc.build(content)
    return target

