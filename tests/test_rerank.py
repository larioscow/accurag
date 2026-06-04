"""Tests for accurag.rerank. All hermetic, without API keys or network."""

from accurag.models import Chunk, RetrievedChunk


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=1,
        text=f"Text for chunk {chunk_id}",
        source_title="Test Paper",
        theme="rag",
        section=None,
        url="https://example.com",
    )


def _make_retrieved(chunk_id: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk=_make_chunk(chunk_id), score=score, rank=rank)


# ---------------------------------------------------------------------------
# Fake Cohere v2 client
# ---------------------------------------------------------------------------


class _FakeRerankResult:
    """Mimics one item in V2RerankResponse.results."""

    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankResponse:
    def __init__(self, results: list[_FakeRerankResult]) -> None:
        self.results = results


class _FakeV2:
    """Mimics the `co.v2` namespace, reversing the document order."""

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> _FakeRerankResponse:
        # Return indices in reversed order, truncated to top_n
        indices = list(range(len(documents) - 1, -1, -1))[:top_n]
        results = [
            _FakeRerankResult(index=idx, relevance_score=1.0 - 0.1 * pos)
            for pos, idx in enumerate(indices)
        ]
        return _FakeRerankResponse(results)


class _FakeClient:
    """Mimics cohere.ClientV2 with v2 attribute."""

    v2 = _FakeV2()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rerank_reverses_order_with_fake_client():
    """With our fake client that reverses order, rerank should flip the list."""
    from accurag.rerank import rerank

    retrieved = [
        _make_retrieved("1-0", score=0.9, rank=0),
        _make_retrieved("1-1", score=0.8, rank=1),
        _make_retrieved("1-2", score=0.7, rank=2),
    ]

    result = rerank("what is RAG?", retrieved, top_n=3, client=_FakeClient())

    # Output order should be reversed (fake always reverses)
    assert [r.chunk.chunk_id for r in result] == ["1-2", "1-1", "1-0"]


def test_rerank_reassigns_ranks_0_indexed():
    """Ranks in output must be 0, 1, 2, … regardless of input ranks."""
    from accurag.rerank import rerank

    retrieved = [
        _make_retrieved("2-0", score=0.9, rank=0),
        _make_retrieved("2-1", score=0.8, rank=1),
        _make_retrieved("2-2", score=0.7, rank=2),
        _make_retrieved("2-3", score=0.6, rank=3),
    ]

    result = rerank("hybrid retrieval", retrieved, top_n=4, client=_FakeClient())

    assert [r.rank for r in result] == [0, 1, 2, 3]


def test_rerank_respects_top_n():
    """top_n truncates the output list."""
    from accurag.rerank import rerank

    retrieved = [
        _make_retrieved("3-0", score=0.9, rank=0),
        _make_retrieved("3-1", score=0.8, rank=1),
        _make_retrieved("3-2", score=0.7, rank=2),
        _make_retrieved("3-3", score=0.6, rank=3),
        _make_retrieved("3-4", score=0.5, rank=4),
    ]

    result = rerank("reranking", retrieved, top_n=2, client=_FakeClient())

    assert len(result) == 2
    # ranks still start from 0
    assert result[0].rank == 0
    assert result[1].rank == 1


def test_rerank_returns_retrieved_chunks_type():
    """Every item in the output must be a RetrievedChunk."""
    from accurag.rerank import rerank

    retrieved = [
        _make_retrieved("4-0", score=0.5, rank=0),
        _make_retrieved("4-1", score=0.4, rank=1),
    ]

    result = rerank("context window", retrieved, top_n=2, client=_FakeClient())

    assert all(isinstance(r, RetrievedChunk) for r in result)


def test_rerank_empty_input_returns_empty():
    """Empty retrieved list returns empty list without error."""
    from accurag.rerank import rerank

    result = rerank("empty", [], top_n=5, client=_FakeClient())

    assert result == []
