"""Lightweight, no-ML document text extraction (PyMuPDF for PDFs, regex strip
for HTML).

This is the fast/low-memory alternative to the Docling parser: pure C text
extraction, no PyTorch, no neural layout models, ~tens of MB of RAM, milliseconds
per doc. Trade-off vs Docling: no structured table extraction (tables come out
as inline text). Used by ``RagPipeline.ingest(parser="pymupdf")``.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n\s*\n\s*\n+")


def _html_to_text(raw: str) -> str:
    raw = _SCRIPT_STYLE.sub(" ", raw)
    raw = re.sub(r"</(p|div|h[1-6]|li|br|tr|section|article)>", "\n", raw, flags=re.IGNORECASE)
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    return _BLANKS.sub("\n\n", text).strip()


def extract_text(path: Path) -> str:
    """Extract plain text from a local PDF or HTML file.

    PDFs are read page-by-page with PyMuPDF; HTML is tag-stripped. Returns the
    document text with blank-line separators between pages/blocks.
    """
    if path.suffix.lower() == ".pdf":
        try:
            import pymupdf
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Parser deps not installed — run: pip install 'accurag[ingest]'"
            ) from exc

        with pymupdf.open(str(path)) as doc:
            pages = [page.get_text("text") for page in doc]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    # HTML / everything else: best-effort tag strip
    return _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
