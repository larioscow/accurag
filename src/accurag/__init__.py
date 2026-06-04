"""accurag: a reusable, evaluation-driven RAG library over a curated corpus.

The public surface is small. Importing this package is cheap and
key-free: ``RagPipeline`` builds its heavy collaborators (anthropic, cohere,
fastembed, qdrant) lazily, so ``import accurag`` needs none of them installed.
"""

from accurag.config import Settings
from accurag.embed import LocalEmbeddingClient
from accurag.llm import EmptyCompletionError
from accurag.models import (
    Answer,
    Chunk,
    EvalReport,
    GoldenQA,
    ManifestEntry,
    RetrievedChunk,
)
from accurag.pipeline import RagPipeline

__all__ = [
    "RagPipeline",
    "Settings",
    "LocalEmbeddingClient",
    "EmptyCompletionError",
    "Answer",
    "RetrievedChunk",
    "Chunk",
    "ManifestEntry",
    "GoldenQA",
    "EvalReport",
]
