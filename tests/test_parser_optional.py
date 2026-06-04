"""The parser deps (docling, pymupdf) live in the optional ``ingest`` extra.

These tests assert that when the underlying lazy import fails with
``ModuleNotFoundError``, the three parser entry points re-raise a friendly
``ImportError`` pointing at the install extra rather than leaking a cryptic
``No module named 'docling'`` to the caller. The deps are still physically
installed in this venv, so we simulate their absence by monkeypatching
``builtins.__import__``.
"""

from __future__ import annotations

import builtins

import pytest

from accurag.models import ManifestEntry

_MSG = "Parser deps not installed. Run: pip install 'accurag[ingest]'"


def _block(monkeypatch, prefix: str) -> None:
    """Make any ``import`` of *prefix* (or a submodule) raise ModuleNotFoundError."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == prefix or name.startswith(prefix + "."):
            raise ModuleNotFoundError(f"No module named '{prefix}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _entry() -> ManifestEntry:
    return ManifestEntry(
        id=1,
        title="T",
        authors="a",
        year=2020,
        arxiv_id=None,
        pdf_url="http://x",
        source="vendor",
        theme="rag",
        has_tables_or_figures=False,
        verified=True,
    )


def test_parse_get_converter_friendly_error(monkeypatch):
    from accurag import parse

    monkeypatch.setattr(parse, "_converter", None)
    _block(monkeypatch, "docling")
    with pytest.raises(ImportError, match="accurag\\[ingest\\]") as exc:
        parse._get_converter()
    assert str(exc.value) == _MSG


def test_chunk_document_friendly_error(monkeypatch):
    from accurag.chunk import chunk_document

    _block(monkeypatch, "docling")
    with pytest.raises(ImportError) as exc:
        chunk_document(object(), _entry())
    assert str(exc.value) == _MSG


def test_fastparse_friendly_error(monkeypatch, tmp_path):
    from accurag.fastparse import extract_text

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    _block(monkeypatch, "pymupdf")
    with pytest.raises(ImportError) as exc:
        extract_text(pdf)
    assert str(exc.value) == _MSG
