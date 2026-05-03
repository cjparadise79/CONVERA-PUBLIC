"""Build CONVERA manuals as simple dependency-free PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parent
DOCUMENTS = [
    ("CONVERA_Project_Report.md", "CONVERA_Project_Report.pdf"),
    ("CONVERA_Command_Manual.md", "CONVERA_Command_Manual.pdf"),
    ("CONVERA_UI_Manual.md", "CONVERA_UI_Manual.pdf"),
]

PAGE_W = 612
PAGE_H = 792
MARGIN_X = 54
MARGIN_TOP = 54
MARGIN_BOTTOM = 54
BODY_SIZE = 10
CODE_SIZE = 8.5


@dataclass
class DrawLine:
    text: str
    font: str
    size: float
    x: float
    y: float


def escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def clean_inline(text: str) -> str:
    return re.sub(r"`([^`]+)`", r"\1", text)


def wrap_text(text: str, *, size: float, width: float, code: bool = False) -> list[str]:
    if not text:
        return [""]
    char_width = size * (0.6 if code else 0.52)
    max_chars = max(18, int(width / char_width))
    if code:
        wrapped: list[str] = []
        for line in text.splitlines() or [""]:
            wrapped.extend(textwrap.wrap(line, width=max_chars, replace_whitespace=False) or [""])
        return wrapped
    return textwrap.wrap(clean_inline(text), width=max_chars) or [""]


def parse_markdown(source: str) -> list[tuple[str, str, float, int]]:
    parsed: list[tuple[str, str, float, int]] = []
    in_code = False
    for raw_line in source.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            parsed.append(("", "F1", BODY_SIZE, 0))
            continue
        if in_code:
            parsed.append((line, "F3", CODE_SIZE, 0))
            continue
        if not line.strip():
            parsed.append(("", "F1", BODY_SIZE, 0))
        elif line.startswith("# "):
            parsed.append((line[2:].strip(), "F2", 20, 0))
        elif line.startswith("## "):
            parsed.append((line[3:].strip(), "F2", 15, 0))
        elif line.startswith("### "):
            parsed.append((line[4:].strip(), "F2", 12, 0))
        elif line.startswith("- "):
            parsed.append(("- " + line[2:].strip(), "F1", BODY_SIZE, 14))
        else:
            parsed.append((line, "F1", BODY_SIZE, 0))
    return parsed


def paginate(items: list[tuple[str, str, float, int]]) -> list[list[DrawLine]]:
    pages: list[list[DrawLine]] = [[]]
    y = PAGE_H - MARGIN_TOP
    usable_width = PAGE_W - (2 * MARGIN_X)

    def new_page() -> None:
        nonlocal y
        pages.append([])
        y = PAGE_H - MARGIN_TOP

    for text, font, size, indent in items:
        leading = size + 4
        if text == "":
            y -= leading * 0.7
            if y < MARGIN_BOTTOM:
                new_page()
            continue

        is_code = font == "F3"
        lines = wrap_text(text, size=size, width=usable_width - indent, code=is_code)
        if font == "F2" and size >= 15 and y < PAGE_H - MARGIN_TOP:
            y -= 5
        for idx, line in enumerate(lines):
            if y < MARGIN_BOTTOM:
                new_page()
            x = MARGIN_X + indent + (14 if idx > 0 and text.startswith("- ") else 0)
            pages[-1].append(DrawLine(line, font, size, x, y))
            y -= leading
        if font == "F2":
            y -= 4
    return pages


def build_pdf(markdown_path: Path, pdf_path: Path) -> None:
    source = markdown_path.read_text(encoding="utf-8")
    pages = paginate(parse_markdown(source))

    objects: list[bytes] = []

    def add_object(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_regular_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_code_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for page in pages:
        commands = []
        for line in page:
            commands.append(
                f"BT /{line.font} {line.size:.2f} Tf {line.x:.2f} {line.y:.2f} Td ({escape_pdf(line.text)}) Tj ET"
            )
        stream = "\n".join(commands).encode("latin-1", errors="replace")
        content_id = add_object(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R /F3 {font_code_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)
        content_ids.append(content_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    pdf = bytearray()
    pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, payload in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(payload)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f\n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n\n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    pdf_path.write_bytes(bytes(pdf))


def main() -> None:
    for markdown_name, pdf_name in DOCUMENTS:
        build_pdf(ROOT / markdown_name, ROOT / pdf_name)
        print(f"wrote {ROOT / pdf_name}")


if __name__ == "__main__":
    main()
