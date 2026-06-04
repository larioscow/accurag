"""Tests for chunk.py — hermetic, no Docling run, no API keys."""

from types import SimpleNamespace

from accurag.chunk import _to_chunk
from accurag.models import Chunk, ManifestEntry


def _make_entry(**kwargs) -> ManifestEntry:
    defaults = {
        "id": 3,
        "title": "Lost in the Middle",
        "authors": "Liu et al.",
        "year": 2023,
        "arxiv_id": "2307.03172",
        "pdf_url": "https://arxiv.org/pdf/2307.03172",
        "source": "arxiv",
        "theme": "rag_foundations",
        "has_tables_or_figures": False,
        "verified": True,
    }
    defaults.update(kwargs)
    return ManifestEntry(**defaults)


def _make_raw_chunk(headings: list[str]) -> SimpleNamespace:
    """Fake Docling BaseChunk with the minimum surface used by _to_chunk."""
    meta = SimpleNamespace(export_json_dict=lambda: {"headings": headings})
    return SimpleNamespace(text="Some chunk text.", meta=meta)


def test_to_chunk_basic_fields():
    entry = _make_entry()
    raw = _make_raw_chunk(["Introduction"])

    chunk = _to_chunk(raw, entry, ordinal=0)

    assert isinstance(chunk, Chunk)
    assert chunk.chunk_id == "3-0"
    assert chunk.doc_id == 3
    assert chunk.text == "Some chunk text."
    assert chunk.source_title == "Lost in the Middle"
    assert chunk.theme == "rag_foundations"
    assert chunk.section == "Introduction"
    assert chunk.url == "https://arxiv.org/pdf/2307.03172"


def test_to_chunk_no_headings_gives_none_section():
    entry = _make_entry()
    raw = _make_raw_chunk([])

    chunk = _to_chunk(raw, entry, ordinal=7)

    assert chunk.chunk_id == "3-7"
    assert chunk.section is None


def test_to_chunk_multiple_headings_takes_first():
    entry = _make_entry()
    raw = _make_raw_chunk(["Chapter 1", "Section 1.1", "Sub-section"])

    chunk = _to_chunk(raw, entry, ordinal=2)

    assert chunk.section == "Chapter 1"


def test_to_chunk_no_arxiv_id_field():
    """Chunk must NOT have an arxiv_id attribute."""
    entry = _make_entry()
    raw = _make_raw_chunk(["Intro"])
    chunk = _to_chunk(raw, entry, ordinal=0)
    assert not hasattr(chunk, "arxiv_id")


def test_chunk_document_is_callable():
    """chunk_document must exist and be callable — import-only check, no Docling run."""
    from accurag.chunk import chunk_document

    assert callable(chunk_document)


def test_chunk_document_drops_empty_chunks(monkeypatch):
    """Whitespace-only chunks are filtered; ordinals stay contiguous (regression:
    OpenAI embeddings reject empty-string inputs)."""
    import docling.chunking as dc

    from accurag.chunk import chunk_document

    def _raw(text):
        return SimpleNamespace(
            text=text, meta=SimpleNamespace(export_json_dict=lambda: {"headings": []})
        )

    class _FakeChunker:
        def chunk(self, doc):
            return [_raw("real one"), _raw("   "), _raw(""), _raw("real two")]

    monkeypatch.setattr(dc, "HybridChunker", lambda *a, **k: _FakeChunker())
    chunks = chunk_document(object(), _make_entry())

    assert [c.text for c in chunks] == ["real one", "real two"]
    assert [c.chunk_id for c in chunks] == ["3-0", "3-1"]  # contiguous, no gap
