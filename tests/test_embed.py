"""Tests for accurag.embed — hermetic, no API keys, no network."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Fake OpenAI client
# ---------------------------------------------------------------------------


def _make_fake_embedding(index: int, dim: int = 4) -> list[float]:
    """Deterministic vector for text at position *index*."""
    return [float(index)] * dim


class _FakeEmbeddingData:
    def __init__(self, index: int) -> None:
        self.embedding = _make_fake_embedding(index)


class _FakeEmbeddingsResponse:
    def __init__(self, indices: list[int]) -> None:
        self.data = [_FakeEmbeddingData(i) for i in indices]


class _FakeEmbeddingsResource:
    """Records every call; returns deterministic vectors keyed by *position*."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingsResponse:
        self.calls.append({"model": model, "input": list(input)})
        # Return index-based vectors so we can verify order is preserved
        n = len(input)
        start = sum(len(c["input"]) for c in self.calls[:-1])
        return _FakeEmbeddingsResponse(list(range(start, start + n)))


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsResource()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embed_texts_returns_one_vector_per_text():
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    texts = ["hello", "world", "foo"]
    result = embed_texts(texts, client=client)

    assert len(result) == 3, "one vector per input text"


def test_embed_texts_order_preserved():
    """Vectors must come back in the same order as the input texts."""
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    texts = [f"doc_{i}" for i in range(7)]
    result = embed_texts(texts, client=client)

    # Each fake vector is [index]*dim — position 0 → [0.0, …], position 3 → [3.0, …]
    for i, vec in enumerate(result):
        assert vec == _make_fake_embedding(i), f"vector at position {i} was out of order: {vec}"


def test_embed_texts_batching():
    """With batch_size=3 and 7 texts, we expect ceil(7/3)=3 API calls."""
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    texts = [f"text_{i}" for i in range(7)]
    embed_texts(texts, client=client, batch_size=3)

    assert len(client.embeddings.calls) == 3


def test_embed_texts_batch_sizes_correct():
    """Batches should be [3, 3, 1] for 7 texts with batch_size=3."""
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    texts = [f"t{i}" for i in range(7)]
    embed_texts(texts, client=client, batch_size=3)

    sizes = [len(c["input"]) for c in client.embeddings.calls]
    assert sizes == [3, 3, 1]


def test_embed_texts_single_text():
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    result = embed_texts(["only one"], client=client)

    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(v, float) for v in result[0])


def test_embed_texts_empty_list():
    from accurag.embed import embed_texts

    client = FakeOpenAIClient()
    result = embed_texts([], client=client)

    assert result == []
    assert len(client.embeddings.calls) == 0


# ---------------------------------------------------------------------------
# Local (no-API) embedding client
# ---------------------------------------------------------------------------


class _FakeFastEmbed:
    """Quacks like fastembed.TextEmbedding: embed(texts) -> iterator of vectors.

    Yields numpy arrays to mirror the real library so the float-coercion path is
    exercised.  Vector encodes the text length so order is verifiable."""

    def embed(self, texts):
        np = __import__("numpy")
        for t in texts:
            yield np.array([len(t), 0, 1], dtype=np.float32)


def test_local_embeddings_adapter_shape_and_native_floats():
    from accurag.embed import _LocalEmbeddings

    emb = _LocalEmbeddings(_FakeFastEmbed())
    resp = emb.create(model="ignored", input=["a", "bb"])
    vecs = [d.embedding for d in resp.data]
    assert vecs == [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
    assert all(type(x) is float for v in vecs for x in v)  # numpy coerced to float


def test_embed_texts_works_with_local_adapter():
    """The local client is a drop-in for embed_texts (same .embeddings.create)."""
    from accurag.embed import _LocalEmbeddings, embed_texts

    class _Client:
        embeddings = _LocalEmbeddings(_FakeFastEmbed())

    vecs = embed_texts(["a", "bb", "ccc"], client=_Client(), batch_size=2)
    assert len(vecs) == 3
    assert vecs[0][0] == 1.0 and vecs[2][0] == 3.0  # order preserved across batches


def test_local_embedding_client_is_exported():
    from accurag.embed import LocalEmbeddingClient

    assert callable(LocalEmbeddingClient)


def test_make_client_is_callable():
    """make_client() must be importable and callable (lazy import — no API key needed
    when the function signature is inspected; we only verify it returns something)."""
    import inspect

    from accurag import embed

    assert callable(embed.make_client)
    sig = inspect.signature(embed.make_client)
    # callable with no args (api_key is optional, defaults to settings)
    assert all(p.default is not inspect.Parameter.empty for p in sig.parameters.values())


def test_embedder_max_tokens_ceiling():
    from accurag.embed import embedder_max_tokens

    class _OpenAILike:  # no max_input_tokens attr -> OpenAI assumption
        pass

    class _Local:
        max_input_tokens = 512

    assert embedder_max_tokens(_OpenAILike()) == 8191
    assert embedder_max_tokens(_Local()) == 512
    # the clamp: min(target, ceiling)
    assert min(800, embedder_max_tokens(_Local())) == 512  # bge caps an 800 target
    assert min(512, embedder_max_tokens(_OpenAILike())) == 512  # OpenAI leaves 512 target alone
