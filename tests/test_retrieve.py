"""Tests for accurag.retrieve: dense and hybrid retrieval.

All tests use QdrantClient(":memory:"), so they need no network, disk, or API keys.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from accurag.models import Chunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Helpers shared between tests
# ---------------------------------------------------------------------------

_DIM = 4  # tiny dimension for in-memory tests
_COLLECTION = "test_col"


def _chunk(chunk_id: str, doc_id: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        source_title="Test Paper",
        theme="rag",
        section="Introduction",
        url="https://example.com",
    )


def _make_dense_client() -> QdrantClient:
    """In-memory Qdrant client pre-populated with two chunks (dense only)."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=_DIM, distance=models.Distance.COSINE)},
    )
    chunks = [_chunk("1-0"), _chunk("1-1")]
    # chunk "1-0" lives near [1, 0, 0, 0]; chunk "1-1" lives near [0, 1, 0, 0]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    client.upsert(
        collection_name=_COLLECTION,
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": vec},
                payload=chunk.model_dump(),
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=False))
        ],
        wait=True,
    )
    return client


def _make_hybrid_client() -> QdrantClient:
    """In-memory Qdrant client pre-populated with two chunks (dense + sparse)."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    chunks = [_chunk("2-0"), _chunk("2-1")]
    dense_vecs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    # chunk "2-0" has sparse token 0 strongly; chunk "2-1" has sparse token 1
    sparse_vecs = [
        models.SparseVector(indices=[0], values=[1.0]),
        models.SparseVector(indices=[1], values=[1.0]),
    ]
    client.upsert(
        collection_name=_COLLECTION,
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": d, "sparse": s},
                payload=chunk.model_dump(),
            )
            for i, (chunk, d, s) in enumerate(zip(chunks, dense_vecs, sparse_vecs, strict=False))
        ],
        wait=True,
    )
    return client


# ---------------------------------------------------------------------------
# Tests for dense()
# ---------------------------------------------------------------------------


def test_dense_returns_list_of_retrieved_chunks():
    from accurag.retrieve import dense

    client = _make_dense_client()
    results = dense(client, query_vec=[1.0, 0.0, 0.0, 0.0], k=2, collection=_COLLECTION)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_dense_nearest_chunk_is_rank_0():
    from accurag.retrieve import dense

    client = _make_dense_client()
    # query close to chunk "1-0"
    results = dense(client, query_vec=[1.0, 0.0, 0.0, 0.0], k=2, collection=_COLLECTION)

    assert results[0].rank == 0
    assert results[1].rank == 1
    assert results[0].chunk.chunk_id == "1-0"


def test_dense_chunk_payload_roundtrips():
    from accurag.retrieve import dense

    client = _make_dense_client()
    results = dense(client, query_vec=[0.0, 1.0, 0.0, 0.0], k=1, collection=_COLLECTION)

    assert results[0].chunk.chunk_id == "1-1"
    assert results[0].chunk.source_title == "Test Paper"


def test_dense_k_limits_results():
    from accurag.retrieve import dense

    client = _make_dense_client()
    results = dense(client, query_vec=[1.0, 0.0, 0.0, 0.0], k=1, collection=_COLLECTION)

    assert len(results) == 1


def test_dense_score_is_float():
    from accurag.retrieve import dense

    client = _make_dense_client()
    results = dense(client, query_vec=[1.0, 0.0, 0.0, 0.0], k=2, collection=_COLLECTION)

    assert all(isinstance(r.score, float) for r in results)


# ---------------------------------------------------------------------------
# Tests for hybrid(): server-side RRF in :memory:
# ---------------------------------------------------------------------------


def test_hybrid_returns_list_of_retrieved_chunks():
    from accurag.retrieve import hybrid

    client = _make_hybrid_client()
    results = hybrid(
        client,
        dense_vec=[1.0, 0.0, 0.0, 0.0],
        sparse_vec=([], []),  # empty sparse = no-op for hybrid
        k=2,
        collection=_COLLECTION,
    )

    assert isinstance(results, list)
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_hybrid_ranks_are_sequential():
    from accurag.retrieve import hybrid

    client = _make_hybrid_client()
    results = hybrid(
        client,
        dense_vec=[1.0, 0.0, 0.0, 0.0],
        sparse_vec=([0], [1.0]),  # favour chunk "2-0"
        k=2,
        collection=_COLLECTION,
    )

    ranks = [r.rank for r in results]
    assert ranks == list(range(len(results)))


def test_hybrid_top_result_has_chunk_id():
    from accurag.retrieve import hybrid

    client = _make_hybrid_client()
    results = hybrid(
        client,
        dense_vec=[1.0, 0.0, 0.0, 0.0],
        sparse_vec=([0], [1.0]),
        k=1,
        collection=_COLLECTION,
    )

    assert results[0].chunk.chunk_id == "2-0"


# ---------------------------------------------------------------------------
# Embedder/index dimension contract. A mismatched embedder must fail loudly.
# ---------------------------------------------------------------------------


def test_dense_raises_on_dim_mismatch():
    from accurag.retrieve import dense

    client = _make_dense_client()  # collection dense dim = 4
    with pytest.raises(ValueError, match="mismatch"):
        dense(client, query_vec=[1.0, 0.0, 0.0], k=1, collection=_COLLECTION)


def test_hybrid_raises_on_dim_mismatch():
    from accurag.retrieve import hybrid

    client = _make_hybrid_client()  # collection dense dim = 4
    with pytest.raises(ValueError, match="mismatch"):
        hybrid(
            client,
            dense_vec=[1.0, 0.0, 0.0],
            sparse_vec=([0], [1.0]),
            k=1,
            collection=_COLLECTION,
        )
