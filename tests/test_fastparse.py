"""Tests for the fast (no-ML) parse + chunk path."""

from __future__ import annotations

from accurag.chunk import chunk_text
from accurag.fastparse import _html_to_text, extract_text
from accurag.models import ManifestEntry


def _entry(**kw) -> ManifestEntry:
    base = {
        "id": 7,
        "title": "T",
        "authors": "a",
        "year": 2020,
        "arxiv_id": None,
        "pdf_url": "http://x",
        "source": "vendor",
        "theme": "rag",
        "has_tables_or_figures": False,
        "verified": True,
    }
    base.update(kw)
    return ManifestEntry(**base)


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>x{}</style></head><body><h1>Title</h1><p>Hello &amp; world</p><script>bad()</script></body></html>"
    out = _html_to_text(html)
    assert "Hello & world" in out
    assert "Title" in out
    assert "bad()" not in out and "<" not in out


def test_extract_text_reads_html_file(tmp_path):
    p = tmp_path / "42.html"
    p.write_text("<p>alpha</p><p>beta</p>")
    out = extract_text(p)
    assert "alpha" in out and "beta" in out


def test_chunk_text_windows_with_overlap():
    text = " ".join(f"word{i}" for i in range(600))  # ~600+ tokens
    chunks = chunk_text(text, _entry(), max_tokens=200, overlap_tokens=40)
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "7-0"
    assert chunks[1].chunk_id == "7-1"
    assert all(c.doc_id == 7 and c.section is None for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_chunk_text_short_input_single_chunk():
    chunks = chunk_text("just a few words here", _entry(), max_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words here"


def test_chunk_text_empty_input():
    assert chunk_text("", _entry()) == []
