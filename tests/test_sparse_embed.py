"""Tests for sparse_embed.py. All hermetic, with no fastembed or network needed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

# ---------------------------------------------------------------------------
# Helpers: a minimal fake SparseEmbedding and fake model that matches the
# fastembed SparseTextEmbedding.embed() contract without importing fastembed.
# ---------------------------------------------------------------------------


def _make_fake_embedding(indices: list[int], values: list[float]) -> Any:
    """Return a SimpleNamespace that quacks like a fastembed SparseEmbedding."""
    return SimpleNamespace(indices=indices, values=values)


class _FakeSparseModel:
    """Fake SparseTextEmbedding: embed() yields one SparseEmbedding per text."""

    def __init__(self, outputs: list[tuple[list[int], list[float]]]) -> None:
        self._outputs = outputs

    def embed(self, texts: list[str]):
        for indices, values in self._outputs:
            yield _make_fake_embedding(indices, values)


# ---------------------------------------------------------------------------
# Import the module under test AFTER helpers so any ImportError is obvious.
# ---------------------------------------------------------------------------


from accurag.sparse_embed import make_sparse_model, sparse_embed_texts  # noqa: E402

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sparse_embed_texts_returns_correct_length():
    """Output list is the same length as input list."""
    fake_model = _FakeSparseModel(
        [
            ([1, 2, 3], [0.5, 0.3, 0.2]),
            ([10, 20], [0.9, 0.1]),
        ]
    )
    result = sparse_embed_texts(["hello world", "foo bar"], model=fake_model)
    assert len(result) == 2


def test_sparse_embed_texts_returns_index_value_tuples():
    """Each item is a (list[int], list[float]) tuple."""
    fake_model = _FakeSparseModel(
        [
            ([7, 42], [0.8, 0.2]),
        ]
    )
    result = sparse_embed_texts(["only one text"], model=fake_model)
    indices, values = result[0]
    assert isinstance(indices, list)
    assert isinstance(values, list)
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)


def test_sparse_embed_texts_preserves_order():
    """Embeddings are aligned to the input order (first text → first tuple)."""
    fake_model = _FakeSparseModel(
        [
            ([1], [0.9]),
            ([2], [0.8]),
            ([3], [0.7]),
        ]
    )
    texts = ["alpha", "beta", "gamma"]
    result = sparse_embed_texts(texts, model=fake_model)
    assert result[0][0] == [1]
    assert result[1][0] == [2]
    assert result[2][0] == [3]


def test_sparse_embed_texts_values_match():
    """Values from the fake embedding appear intact in the output."""
    fake_model = _FakeSparseModel(
        [
            ([100, 200], [0.11, 0.22]),
        ]
    )
    result = sparse_embed_texts(["test"], model=fake_model)
    indices, values = result[0]
    assert indices == [100, 200]
    assert values == [0.11, 0.22]


def test_sparse_embed_texts_empty_input():
    """An empty input list returns an empty output list."""
    fake_model = _FakeSparseModel([])
    result = sparse_embed_texts([], model=fake_model)
    assert result == []


def test_sparse_embed_texts_coerces_numpy_to_native_python():
    """fastembed yields numpy int64/float32; output must be native int/float so
    it is JSON-serialisable (regression: pipeline persists these to chunks.jsonl)."""
    import json

    np = __import__("numpy")
    fake_model = _FakeSparseModel(
        [
            (np.array([5, 9], dtype=np.int64), np.array([0.4, 0.6], dtype=np.float32)),
        ]
    )
    indices, values = sparse_embed_texts(["x"], model=fake_model)[0]
    assert all(type(i) is int for i in indices)
    assert all(type(v) is float for v in values)
    # The whole record must round-trip through json without error.
    json.dumps({"indices": indices, "values": values})


def test_make_sparse_model_is_callable():
    """make_sparse_model is a callable (we only check the symbol exists and is
    callable; we do NOT invoke it so no network/fastembed download is needed)."""
    assert callable(make_sparse_model)
