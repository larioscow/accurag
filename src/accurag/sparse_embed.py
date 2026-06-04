"""Sparse text embedding via fastembed SparseTextEmbedding (BM25/SPLADE).

The fastembed SDK is imported *lazily* inside make_sparse_model() so the
module can be imported and unit-tested without fastembed installed or any
network access.  Pass an injectable ``model`` to sparse_embed_texts() in
tests (see tests/test_sparse_embed.py).
"""

from __future__ import annotations

from typing import Any

# Default sparse model — SPLADE++ is the primary fastembed sparse model and
# serves the BM25/BM42-style sparse-representation role in the pipeline.
_DEFAULT_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"


def make_sparse_model(model_name: str = _DEFAULT_SPARSE_MODEL) -> Any:
    """Lazily build and return a fastembed SparseTextEmbedding model.

    The fastembed SDK is imported inside this function so the module can be
    imported without fastembed being installed (unit tests inject a fake model
    and never call this function).
    """
    from fastembed import SparseTextEmbedding  # lazy import

    return SparseTextEmbedding(model_name=model_name)


def sparse_embed_texts(
    texts: list[str],
    model: Any = None,
    batch_size: int = 256,
) -> list[tuple[list[int], list[float]]]:
    """Return sparse embeddings for *texts*, preserving order.

    Each element of the returned list is a ``(indices, values)`` tuple
    matching the fastembed ``SparseEmbedding`` contract.

    Args:
        texts:  Input strings to embed.
        model:  An object with an ``embed(texts) -> Iterable[SparseEmbedding]``
                method.  Defaults to ``make_sparse_model()`` when *None*.
                Inject a fake in unit tests so fastembed is never needed.

    Returns:
        A list of ``(indices, values)`` tuples, one per input text, in the
        same order as *texts*.
    """
    if not texts:
        return []

    if model is None:
        model = make_sparse_model()

    results: list[tuple[list[int], list[float]]] = []
    # Batch the input. fastembed runs a native onnx model and parallelises large
    # inputs, which segfaults at full-corpus scale (thousands of texts) on some
    # platforms (the leaked-semaphore crash on Apple Silicon). Keeping each
    # embed() call small — mirroring the dense embedder's batching — avoids that
    # and bounds memory. Order is preserved across batches.
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for embedding in model.embed(batch):
            # fastembed yields numpy int64/float32 arrays; coerce to native
            # Python int/float so the result honours the typed contract and
            # stays JSON-serialisable (the pipeline persists these to jsonl).
            indices: list[int] = [int(i) for i in embedding.indices]
            values: list[float] = [float(v) for v in embedding.values]
            results.append((indices, values))

    return results
