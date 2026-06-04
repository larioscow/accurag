"""Qdrant index helpers: create collection and upsert chunks.

Design notes
------------
* make_client() is the production entry-point; tests inject their own
  QdrantClient(":memory:") to stay hermetic.
* Named vectors: "dense" (dense float vectors) and optionally "sparse"
  (BM25/BM42 SparseVector). This mirrors the hybrid-retrieval plan.
* Qdrant client 1.18 uses create_collection / collection_exists — the old
  recreate_collection is gone.
* Point IDs: Qdrant requires integer or UUID point IDs. We derive a stable
  unsigned 64-bit integer from the chunk_id string via SHA-256 so that
  chunk_id="1-0" becomes a repeatable numeric ID. The human-readable
  chunk_id is preserved in the payload (via chunk.model_dump()).
* Lazy import: QdrantClient is imported at call-time inside make_client() so
  the module can be imported in environments where qdrant-client is absent
  (not currently a concern, but mirrors the project convention).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

from accurag.config import settings
from accurag.models import Chunk


def dense_dim_of(client: QdrantClient, col: str) -> int:
    """Return the size of a collection's ``"dense"`` named vector.

    Single source of truth for "what dim is this collection?" — used by both the
    reuse guard in :func:`build_collection` and the query-time check in
    :func:`accurag.retrieve._assert_dense_dim`, so the two never drift. Handles
    both the named-vector dict shape and a bare (unnamed) VectorParams.
    """
    vectors = client.get_collection(col).config.params.vectors
    if isinstance(vectors, dict):
        if "dense" not in vectors:
            raise ValueError(
                f"Collection '{col}' has named vectors {sorted(vectors)} but no "
                "'dense' vector — it was not built by accurag."
            )
        return vectors["dense"].size
    if vectors is None:
        raise ValueError(
            f"Collection '{col}' has no dense vector configuration — it was not built by accurag."
        )
    return vectors.size


def _sparse_names_of(client: QdrantClient, col: str) -> set[str]:
    """Return the names of a collection's sparse vectors (empty set if none)."""
    sparse = client.get_collection(col).config.params.sparse_vectors
    return set(sparse or {})


def _chunk_id_to_int(chunk_id: str) -> int:
    """Derive a stable unsigned 64-bit integer from a chunk_id string.

    Qdrant point IDs must be unsigned integers or UUIDs. We hash the
    chunk_id with SHA-256 and take the first 8 bytes so that each unique
    chunk_id maps to a unique, repeatable numeric ID.
    """
    digest = hashlib.sha256(chunk_id.encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def make_client(path=None) -> QdrantClient:
    """Return a persistent QdrantClient backed by *path* (default settings.qdrant_path)."""
    from qdrant_client import QdrantClient  # lazy import

    path = path if path is not None else settings.qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def build_collection(
    client: QdrantClient,
    dim: int | None = None,
    *,
    with_sparse: bool = True,
    collection: str | None = None,
) -> None:
    """Create the collection if it does not already exist.

    Parameters
    ----------
    client:
        An injected QdrantClient (may be in-memory for tests).
    dim:
        Dense vector dimension. Defaults to settings.embed_dim (3072 for
        text-embedding-3-large).
    with_sparse:
        Whether to add a named "sparse" vector slot for BM25/BM42 hybrid
        retrieval. Defaults to True.
    collection:
        Collection name; defaults to ``settings.collection``.
    """
    from qdrant_client import models  # lazy import

    col = collection if collection is not None else settings.collection
    vector_size = dim if dim is not None else settings.embed_dim

    if client.collection_exists(col):
        # Idempotent — but guard against silently reusing a collection that does
        # not match what this embedder/strategy needs, which would otherwise fail
        # cryptically deep in Qdrant on the first upsert/query. Two axes:
        #   1. dense dim (e.g. 3072-dim large vs 1536-dim small), and
        #   2. the sparse slot (a dense-only collection reused for hybrid has no
        #      "sparse" vector -> "vector named sparse does not exist").
        existing_dim = dense_dim_of(client, col)
        if existing_dim != vector_size:
            raise ValueError(
                f"Collection '{col}' already exists with dense dim {existing_dim}, "
                f"but this embedder produces {vector_size}-dim vectors. Use a "
                f"different collection name (e.g. RagPipeline(collection=...)) or "
                f"delete the existing collection before re-indexing."
            )
        if with_sparse and "sparse" not in _sparse_names_of(client, col):
            raise ValueError(
                f"Collection '{col}' already exists without a 'sparse' vector, but "
                f"hybrid retrieval needs one. Use a different collection name or "
                f"delete the existing collection before re-indexing."
            )
        return

    vectors_config = {
        "dense": models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        )
    }

    sparse_vectors_config: dict | None = (
        {"sparse": models.SparseVectorParams()} if with_sparse else None
    )

    client.create_collection(
        collection_name=col,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[tuple[list[int], list[float]]] | None = None,
    collection: str | None = None,
) -> None:
    """Upsert chunks into Qdrant.

    Parameters
    ----------
    client:
        An injected QdrantClient.
    chunks:
        Chunk objects whose ``chunk_id`` is used as the point ID.
    dense_vectors:
        One dense float vector per chunk, same order as ``chunks``.
    sparse_vectors:
        Optional list of ``(indices, values)`` tuples from sparse_embed_texts.
        When provided, stored under the "sparse" named vector.
    """
    from qdrant_client import models  # lazy import

    points: list[models.PointStruct] = []
    for i, (chunk, dense_vec) in enumerate(zip(chunks, dense_vectors, strict=True)):
        vector: dict = {"dense": dense_vec}

        if sparse_vectors is not None:
            indices, values = sparse_vectors[i]
            vector["sparse"] = models.SparseVector(indices=indices, values=values)

        points.append(
            models.PointStruct(
                id=_chunk_id_to_int(chunk.chunk_id),
                vector=vector,
                payload=chunk.model_dump(),
            )
        )

    if points:
        client.upsert(
            collection_name=collection if collection is not None else settings.collection,
            points=points,
            wait=True,
        )
