"""Retrieval from Qdrant: dense and hybrid (dense+sparse, server-side RRF).

All public functions accept an injectable QdrantClient so callers and tests
can pass any client (e.g. QdrantClient(":memory:")) without touching disk or
the network.

API verified against qdrant-client current docs via Context7:
- Dense: query_points(collection_name, query=[...], using="dense", limit=k, with_payload=True)
- Hybrid: query_points with prefetch=[Prefetch(dense), Prefetch(sparse)]
          and query=FusionQuery(fusion=Fusion.RRF)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from accurag.config import settings
from accurag.models import Chunk, RetrievedChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


def _assert_dense_dim(client: QdrantClient, col: str, query_vec: list[float]) -> None:
    """Fail loudly if *query_vec* dim doesn't match the collection's dense vector size.

    Guards against a mismatched query embedder (e.g. OpenAI=3072) being run
    against an index built by a different embedder (e.g. bge-small=384).
    """
    from accurag.index import dense_dim_of

    size = dense_dim_of(client, col)
    if len(query_vec) != size:
        raise ValueError(
            f"Embedder/index mismatch: query vector dim {len(query_vec)} != "
            f"collection '{col}' dense dim {size}. "
            "Ingest and query must use the same embedder."
        )


def dense(
    client: QdrantClient,
    query_vec: list[float],
    k: int = 5,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks nearest to *query_vec* using dense vector search.

    Parameters
    ----------
    client:
        An already-constructed QdrantClient (injectable for testing).
    query_vec:
        The dense query embedding.
    k:
        Maximum number of results to return.
    collection:
        Collection name; defaults to ``settings.collection``.

    Returns
    -------
    list[RetrievedChunk]
        Results ordered by descending score, rank 0 … k-1.
    """
    col = collection or settings.collection
    _assert_dense_dim(client, col, query_vec)
    result = client.query_points(
        collection_name=col,
        query=query_vec,
        using="dense",
        limit=k,
        with_payload=True,
    )
    return [
        RetrievedChunk(
            chunk=Chunk(**(point.payload or {})),
            score=float(point.score),
            rank=i,
        )
        for i, point in enumerate(result.points)
    ]


def hybrid(
    client: QdrantClient,
    dense_vec: list[float],
    sparse_vec: tuple[list[int], list[float]],
    k: int = 5,
    collection: str | None = None,
    prefetch_limit: int = 50,
) -> list[RetrievedChunk]:
    """Return the top-k chunks using hybrid retrieval (dense + sparse, RRF fusion).

    Uses Qdrant's server-side multi-stage query with two prefetch branches fused
    by Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    client:
        An already-constructed QdrantClient (injectable for testing).
    dense_vec:
        The dense query embedding.
    sparse_vec:
        A ``(indices, values)`` tuple representing the sparse query vector.
    k:
        Maximum number of results to return after fusion.
    collection:
        Collection name; defaults to ``settings.collection``.
    prefetch_limit:
        Number of candidates each prefetch branch fetches before fusion.

    Returns
    -------
    list[RetrievedChunk]
        Results ordered by descending fused score, rank 0 … k-1.
    """
    from qdrant_client import models

    col = collection or settings.collection
    _assert_dense_dim(client, col, dense_vec)
    indices, values = sparse_vec

    result = client.query_points(
        collection_name=col,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return [
        RetrievedChunk(
            chunk=Chunk(**(point.payload or {})),
            score=float(point.score),
            rank=i,
        )
        for i, point in enumerate(result.points)
    ]
