"""Structure-aware chunking using Docling's HybridChunker.

Docling is imported lazily so tests run without it installed (or with a fake).
The public surface is:
    _to_chunk(raw_chunk, entry, ordinal) -> Chunk
    chunk_document(doc, entry) -> list[Chunk]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from accurag.models import Chunk, ManifestEntry

if TYPE_CHECKING:
    # Only for type-checker; not evaluated at runtime.
    from docling.chunking import HybridChunker as _HybridChunker  # noqa: F401


def _section_from_meta(raw_chunk: Any) -> str | None:
    """Return the first heading from chunk.meta.export_json_dict(), or None."""
    try:
        headings: list[str] = raw_chunk.meta.export_json_dict().get("headings", [])
        return headings[0] if headings else None
    except Exception:  # noqa: BLE001 (section is best-effort; odd/missing Docling)
        return None  # heading metadata legitimately maps to no section


def _to_chunk(raw_chunk: Any, entry: ManifestEntry, ordinal: int) -> Chunk:
    """Convert a single Docling BaseChunk + manifest entry into a Chunk.

    Args:
        raw_chunk: A Docling BaseChunk (or any compatible fake in tests).
        entry:     The ManifestEntry the document came from.
        ordinal:   Zero-based index of this chunk within the document.

    Returns:
        A Chunk with chunk_id = f"{entry.id}-{ordinal}".
    """
    return Chunk(
        chunk_id=f"{entry.id}-{ordinal}",
        doc_id=entry.id,
        text=raw_chunk.text,
        source_title=entry.title,
        theme=entry.theme,
        section=_section_from_meta(raw_chunk),
        url=entry.pdf_url,
    )


def _make_hybrid_chunker(max_tokens: int | None):
    """Build a Docling HybridChunker, optionally with a custom token budget.

    The budget lives on the chunker's tokenizer (``HuggingFaceTokenizer.max_tokens``),
    so when ``max_tokens`` is given we re-wrap the default chunker's tokenizer with
    the new budget. ``None`` uses Docling's default and keeps the simple monkeypatch path.
    """
    try:
        from docling.chunking import HybridChunker  # lazy import
    except ModuleNotFoundError as exc:
        raise ImportError("Parser deps not installed. Run: pip install 'accurag[ingest]'") from exc

    if not max_tokens:
        return HybridChunker()

    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    base = HybridChunker()
    tok = HuggingFaceTokenizer(tokenizer=base.tokenizer.get_tokenizer(), max_tokens=max_tokens)
    return HybridChunker(tokenizer=tok)


def chunk_document(doc: Any, entry: ManifestEntry, max_tokens: int | None = None) -> list[Chunk]:
    """Chunk a DoclingDocument into a list of Chunk objects.

    Uses Docling's HybridChunker (structure-aware), imported lazily. ``max_tokens``
    sets the per-chunk token budget; when None, Docling's default (512) is used.

    Args:
        doc:        A DoclingDocument returned by parse.parse_file().
        entry:      The ManifestEntry describing the source document.
        max_tokens: Target chunk-size budget (already clamped to the embedder).

    Returns:
        Ordered list of Chunk objects, one per HybridChunker output chunk.
    """
    chunker = _make_hybrid_chunker(max_tokens)
    # Skip empty / whitespace-only chunks: they carry no retrievable signal and
    # the OpenAI embeddings API rejects empty-string inputs (400). Ordinals are
    # assigned after filtering so chunk_ids stay contiguous.
    kept = (rc for rc in chunker.chunk(doc) if (rc.text or "").strip())
    return [_to_chunk(rc, entry, ordinal) for ordinal, rc in enumerate(kept)]


def chunk_text(
    text: str,
    entry: ManifestEntry,
    max_tokens: int = 500,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """Chunk raw extracted text into overlapping token windows.

    The lightweight counterpart to :func:`chunk_document` for the no-ML
    ``fastparse`` path: a sliding window over the document's tokens
    (~``max_tokens`` each, ``overlap_tokens`` overlap) using the same tiktoken
    encoding as the embedder. There is no heading/section metadata (section=None).
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    step = max(1, max_tokens - overlap_tokens)

    chunks: list[Chunk] = []
    ordinal = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        body = enc.decode(window).strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{entry.id}-{ordinal}",
                doc_id=entry.id,
                text=body,
                source_title=entry.title,
                theme=entry.theme,
                section=None,
                url=entry.pdf_url,
            )
        )
        ordinal += 1
        if start + max_tokens >= len(tokens):
            break
    return chunks
