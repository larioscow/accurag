"""Tests for src/accurag/index.py. Hermetic, in-memory Qdrant only."""

import pytest
from qdrant_client import QdrantClient

from accurag.index import build_collection, index_chunks
from accurag.models import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # tiny dimension so tests are fast


def _make_chunk(chunk_id: str, doc_id: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        source_title="Test Paper",
        theme="rag",
        section="Introduction",
        url="https://example.com/paper.pdf",
    )


def _vec(val: float) -> list[float]:
    return [val] * DIM


# ---------------------------------------------------------------------------
# build_collection
# ---------------------------------------------------------------------------


def test_build_collection_dense_only_creates_collection():
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False)

    from accurag.config import settings

    assert client.collection_exists(settings.collection)


def test_build_collection_with_sparse_creates_collection():
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=True)

    from accurag.config import settings

    assert client.collection_exists(settings.collection)


def test_build_collection_idempotent_does_not_raise():
    """Calling build_collection twice must not raise."""
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False)
    build_collection(client, dim=DIM, with_sparse=False)  # second call should be a no-op

    from accurag.config import settings

    assert client.collection_exists(settings.collection)


def test_build_collection_dim_mismatch_raises_clearly():
    """Reusing a collection with a different embedder dim must raise rather than
    silently upsert wrong-sized vectors (e.g. 1536-dim small into a 3072 box)."""
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False, collection="c")
    with pytest.raises(ValueError, match="already exists with dense dim"):
        build_collection(client, dim=DIM + 1, with_sparse=False, collection="c")


def test_build_collection_reuse_dense_only_for_hybrid_raises():
    """A dense-only collection reused for hybrid must fail clearly here, before it
    surfaces as an opaque 'vector named sparse does not exist' deep in Qdrant."""
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False, collection="d")
    with pytest.raises(ValueError, match="without a 'sparse' vector"):
        build_collection(client, dim=DIM, with_sparse=True, collection="d")


def test_dense_dim_of_reads_size_and_rejects_non_accurag_collection():
    from qdrant_client import models

    from accurag.index import dense_dim_of

    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False, collection="e")
    assert dense_dim_of(client, "e") == DIM

    # a named-vector collection without a "dense" vector is not ours, so it errors
    client.create_collection(
        collection_name="weird",
        vectors_config={"other": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
    )
    with pytest.raises(ValueError, match=r"no\s+'dense' vector"):
        dense_dim_of(client, "weird")


# ---------------------------------------------------------------------------
# index_chunks (dense only)
# ---------------------------------------------------------------------------


def test_index_chunks_dense_only_correct_count():
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False)

    chunks = [_make_chunk("1-0"), _make_chunk("1-1")]
    dense_vectors = [_vec(0.1), _vec(0.9)]

    index_chunks(client, chunks, dense_vectors, sparse_vectors=None)

    from accurag.config import settings

    result = client.count(collection_name=settings.collection, exact=True)
    assert result.count == 2


def test_index_chunks_payload_round_trips():
    """Payload stored must match the Chunk's model_dump()."""
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False)

    chunk = _make_chunk("2-0", doc_id=2)
    index_chunks(client, [chunk], [_vec(0.5)])

    from accurag.config import settings

    # scroll all points to inspect payload
    points, _ = client.scroll(
        collection_name=settings.collection,
        with_payload=True,
        with_vectors=False,
        limit=10,
    )
    assert len(points) == 1
    assert points[0].payload == chunk.model_dump()


def test_index_chunks_dense_only_two_chunks_then_one_more():
    """Calling index_chunks twice appends, so the total becomes 3."""
    client = QdrantClient(":memory:")
    build_collection(client, dim=DIM, with_sparse=False)

    index_chunks(client, [_make_chunk("1-0"), _make_chunk("1-1")], [_vec(0.1), _vec(0.2)])
    index_chunks(client, [_make_chunk("1-2")], [_vec(0.3)])

    from accurag.config import settings

    assert client.count(collection_name=settings.collection, exact=True).count == 3
