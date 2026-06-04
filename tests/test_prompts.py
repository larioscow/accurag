"""Tests for accurag.prompts — pure Python, no external dependencies."""

from __future__ import annotations

from accurag.models import Chunk, RetrievedChunk
from accurag.prompts import PROMPT_VERSION, build_grounded_prompt


def _make_chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=1,
        text=text,
        source_title="Test Paper",
        theme="rag",
        section="Introduction",
        url="https://example.com/paper",
    )


def _make_retrieved(chunk_id: str, text: str = "some text", rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=_make_chunk(chunk_id, text), score=0.9, rank=rank)


def test_prompt_version_is_v2():
    assert PROMPT_VERSION == "v2"


def test_prompt_contains_query():
    query = "What is retrieval-augmented generation?"
    retrieved = [_make_retrieved("1-0")]
    prompt = build_grounded_prompt(query, retrieved)
    assert query in prompt


def test_prompt_contains_all_chunk_ids():
    query = "How does hybrid retrieval work?"
    retrieved = [
        _make_retrieved("1-0", rank=0),
        _make_retrieved("2-3", rank=1),
        _make_retrieved("5-7", rank=2),
    ]
    prompt = build_grounded_prompt(query, retrieved)
    for rc in retrieved:
        assert rc.chunk.chunk_id in prompt


def test_prompt_contains_i_dont_know_guardrail():
    """The prompt must instruct the model to say 'I don't know' when the
    answer is not in the context — the core anti-hallucination guard."""
    query = "Anything"
    retrieved = [_make_retrieved("1-0")]
    prompt = build_grounded_prompt(query, retrieved)
    # The guardrail phrasing must appear verbatim (case-insensitive is fine,
    # but check for the key sentinel).
    assert "i don't know" in prompt.lower() or "i do not know" in prompt.lower()


def test_prompt_instructs_answer_only_from_context():
    """Must tell the model to answer ONLY from the provided context."""
    query = "anything"
    retrieved = [_make_retrieved("1-0")]
    prompt = build_grounded_prompt(query, retrieved)
    low = prompt.lower()
    # Look for the critical "only" + "context" instruction
    assert "only" in low and "context" in low


def test_prompt_instructs_clean_prose_no_citation_markers():
    """The model must write clean prose — sources are tracked separately, so the
    prompt tells it NOT to add citation/chunk-id markers."""
    query = "How does reranking help?"
    retrieved = [_make_retrieved("3-1")]
    prompt = build_grounded_prompt(query, retrieved)
    low = prompt.lower()
    assert "do not add citation" in low or "no citation markers" in low


def test_prompt_numbers_context_blocks():
    """Each context block should be numbered [1], [2], [3] etc."""
    query = "test"
    retrieved = [
        _make_retrieved("a-0", rank=0),
        _make_retrieved("b-0", rank=1),
        _make_retrieved("c-0", rank=2),
    ]
    prompt = build_grounded_prompt(query, retrieved)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "[3]" in prompt


def test_prompt_includes_source_title():
    """Each context block must expose source_title for traceability."""
    query = "anything"
    retrieved = [_make_retrieved("1-0")]
    prompt = build_grounded_prompt(query, retrieved)
    assert "Test Paper" in prompt


def test_prompt_includes_chunk_text():
    """The actual chunk text must appear in the prompt."""
    query = "anything"
    retrieved = [_make_retrieved("1-0", text="Hybrid retrieval combines dense and sparse.")]
    prompt = build_grounded_prompt(query, retrieved)
    assert "Hybrid retrieval combines dense and sparse." in prompt


def test_empty_retrieved_still_returns_prompt():
    """When no chunks are retrieved the prompt must still be valid and
    include the guardrail instruction."""
    query = "What is RAG?"
    prompt = build_grounded_prompt(query, [])
    assert query in prompt
    low = prompt.lower()
    assert "i don't know" in low or "i do not know" in low
