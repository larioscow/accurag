"""Reranking via Cohere's v2 rerank API.

Design notes
------------
- `cohere` is imported **lazily** (inside `_make_client`) so the module can be
  imported with no API key and no network — satisfying the unit-test requirement.
- The underlying Cohere client is **injectable** via the `client` parameter, so
  tests pass a fake without ever touching the SDK.
- Uses `co.v2.rerank(...)` which is the current (non-deprecated) call shape.
  Response items carry `.index` (original position) and `.relevance_score`.
"""

from __future__ import annotations

from typing import Any

from accurag.models import RetrievedChunk


def _make_client(api_key: str | None = None) -> Any:
    """Lazily import cohere and build a ClientV2 from *api_key* (default settings)."""
    import cohere  # lazy — keeps import cost + key requirement out of module load

    from accurag.config import settings

    key = api_key if api_key is not None else settings.cohere_api_key
    # NOTE: pass the key as-is — do NOT add the `or None` coercion used by
    # embed.py / llm.py. cohere.ClientV2 *raises* on api_key=None but tolerates
    # "" (failure deferred to the first call). "Unifying" the three providers
    # with `or None` would turn cohere's unconfigured case into an eager crash.
    return cohere.ClientV2(api_key=key)


def _is_rate_limit(exc: Exception) -> bool:
    """True if *exc* looks like an HTTP 429 / rate-limit error (provider-agnostic)."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "toomanyrequests" in name or "429" in text or "rate limit" in text


def _rerank_with_retry(
    client: Any,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
    max_retries: int = 6,
    wait_seconds: float = 7.0,
) -> Any:
    """Call Cohere rerank, retrying on rate-limit (429) with a fixed backoff.

    Cohere Trial keys cap at ~10 calls/min, so an eval sweep trips 429s; a short
    wait keeps the call under the limit instead of crashing the whole run.
    """
    import time

    for attempt in range(max_retries):
        try:
            return client.v2.rerank(model=model, query=query, documents=documents, top_n=top_n)
        except Exception as exc:
            if _is_rate_limit(exc) and attempt < max_retries - 1:
                time.sleep(wait_seconds)
                continue
            raise


def rerank(
    query: str,
    retrieved: list[RetrievedChunk],
    top_n: int = 5,
    client: Any | None = None,
    model: str = "rerank-v4.0-pro",
) -> list[RetrievedChunk]:
    """Reorder *retrieved* by relevance to *query* using Cohere rerank v2.

    Parameters
    ----------
    query:
        The user query string.
    retrieved:
        Chunks coming out of dense or hybrid retrieval.
    top_n:
        Maximum number of results to return.
    client:
        An injectable Cohere-compatible client (must expose a `.v2.rerank`
        method).  When ``None`` a live ``cohere.ClientV2`` is built from
        ``COHERE_API_KEY``.
    model:
        Cohere rerank model name.

    Returns
    -------
    list[RetrievedChunk]
        At most *top_n* items, sorted by descending relevance, with
        ``.rank`` reassigned to 0-based positions.
    """
    if not retrieved:
        return []

    if client is None:
        client = _make_client()

    # Cohere v2 rerank expects plain strings
    documents: list[str] = [rc.chunk.text for rc in retrieved]

    response = _rerank_with_retry(client, model, query, documents, min(top_n, len(documents)))

    # response.results is sorted by descending relevance_score;
    # each result carries `.index` — the position in the original `retrieved` list.
    reranked: list[RetrievedChunk] = []
    for new_rank, result in enumerate(response.results):
        original = retrieved[result.index]
        reranked.append(
            RetrievedChunk(
                chunk=original.chunk,
                score=result.relevance_score,
                rank=new_rank,
            )
        )

    return reranked
