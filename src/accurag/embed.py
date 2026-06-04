"""Dense embedding via the OpenAI Embeddings API.

- ``make_client()`` lazily imports ``openai`` so the module can be imported
  without an API key present (unit tests inject a fake client directly).
- ``embed_texts()`` accepts an injectable client parameter for the same reason.
- Batching is order-preserving: results are concatenated in input order.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from accurag.config import settings


def make_client(api_key: str | None = None) -> Any:
    """Return a fresh ``openai.OpenAI`` client using *api_key* (default settings).

    The import is deferred so that importing this module never triggers the
    OpenAI SDK import (which fails in environments without the package or
    without a key set).
    """
    import openai

    key = api_key if api_key is not None else settings.openai_api_key
    # An empty key collapses to None so the OpenAI SDK applies its standard
    # resolution (the ambient OPENAI_API_KEY env var). An unset per-instance
    # key then falls back to the environment rather than failing, which matches
    # the SDK's own behaviour.
    return openai.OpenAI(api_key=key or None)


def embed_texts(
    texts: list[str],
    client: Any,
    batch_size: int = 100,
    model: str | None = None,
) -> list[list[float]]:
    """Embed *texts* using the OpenAI Embeddings API.

    Args:
        texts:      Input strings to embed. May be empty.
        client:     An OpenAI client (real or fake) with an
                    ``embeddings.create(model=..., input=...)`` method.
        batch_size: Maximum number of texts per API call.  The API has an
                    upper limit of 2 048 inputs per request; keep this at or
                    below that.  Defaults to 100.

    Returns:
        A list of float vectors, in the same order as *texts*.
        Each vector has length ``settings.embed_dim``.
    """
    if not texts:
        return []

    embed_model = model if model is not None else settings.embed_model
    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=embed_model,
            input=batch,
        )
        # response.data is ordered to match the input batch
        vectors.extend(item.embedding for item in response.data)

    return vectors


# OpenAI text-embedding-3-large / -small / ada-002 all accept an 8191-token input.
_OPENAI_EMBED_MAX_TOKENS = 8191
# fastembed model -> input-token window (the chunk-budget ceiling).
_LOCAL_EMBED_MAX_TOKENS = {"BAAI/bge-small-en-v1.5": 512}


def embedder_max_tokens(client: Any) -> int:
    """Return the embedder's max input tokens, the chunk-size ceiling.

    A chunk longer than this is silently truncated by the model, so the chunker
    must clamp to it. A client may declare ``max_input_tokens`` (the local
    client does); otherwise we assume an OpenAI embedder (8191-token window).
    """
    declared = getattr(client, "max_input_tokens", None)
    return int(declared) if declared else _OPENAI_EMBED_MAX_TOKENS


# ---------------------------------------------------------------------------
# Local (no-API) embeddings: runs fully offline via fastembed.
# ---------------------------------------------------------------------------


class _LocalEmbeddings:
    """Adapter exposing the slice of the OpenAI embeddings API that
    :func:`embed_texts` uses (``create(model, input) -> obj.data[i].embedding``),
    backed by a local fastembed model. Coerces numpy scalars to native floats."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def create(self, model: str, input: list[str]) -> Any:  # noqa: A002 - match OpenAI sig
        # NOTE: ``model`` is accepted for OpenAI-signature compatibility but
        # ignored. The fastembed model is fixed at construction; swapping it
        # would reload a heavy model per call. The owning LocalEmbeddingClient
        # exposes the real model via ``.model_name``.
        vectors = [[float(x) for x in vec] for vec in self._model.embed(list(input))]
        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


class LocalEmbeddingClient:
    """OpenAI-embeddings-compatible client backed by a local fastembed model.

    Substitutes for the OpenAI client in :func:`embed_texts` and for
    ``RagPipeline(embed_client=...)``, letting ingestion and retrieval run with
    no API key and no network access (after the model is cached on first use).
    Default model ``BAAI/bge-small-en-v1.5`` (384-dim).

    ``fastembed`` is imported lazily so importing ``accurag`` never requires it.

    Note: the embedding dimension differs from OpenAI's, so an index built with
    this client must also be queried with it. Keep the embedder consistent
    across ingest and retrieve.

    The model is chosen here, at construction. ``RagPipeline``'s ``embed_model``
    setting targets the OpenAI embedder and does not apply to this client. Read
    ``.model_name`` to see which local model is actually in use.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from fastembed import TextEmbedding

        name = model_name or settings.local_embed_model
        self.model_name = name
        self.max_input_tokens = _LOCAL_EMBED_MAX_TOKENS.get(name, 512)
        self.embeddings = _LocalEmbeddings(TextEmbedding(model_name=name))
