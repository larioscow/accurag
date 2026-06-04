"""Tests for accurag.pipeline.RagPipeline — fully hermetic.

All collaborators (embed client, sparse model, llm, Qdrant) are FAKE or
in-memory, so these tests run with NO API keys and NO network.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient, models

from accurag.config import settings
from accurag.models import (
    Answer,
    Chunk,
    EvalReport,
    GoldenQA,
    RetrievedChunk,
)
from accurag.pipeline import RagPipeline

_DIM = 4


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEmbedClient:
    """Mimics openai.OpenAI().embeddings.create(model=, input=).

    Returns a deterministic dim-4 vector per text so retrieval is predictable.
    """

    class _Embeddings:
        def create(self, model: str, input: list[str]):
            class _Item:
                def __init__(self, embedding):
                    self.embedding = embedding

            class _Resp:
                def __init__(self, data):
                    self.data = data

            # map a text to a vector: "1-0" -> axis 0, "1-1" -> axis 1, etc.
            data = [_Item(_vec_for(text)) for text in input]
            return _Resp(data)

    def __init__(self):
        self.embeddings = self._Embeddings()


def _axis_for(text: str) -> int:
    """Axis keyed off the LAST digit in text so '1-0' and '1-1' differ."""
    axis = 0
    for ch in text:
        if ch.isdigit():
            axis = int(ch) % _DIM
    return axis


def _vec_for(text: str) -> list[float]:
    """Deterministic dim-4 one-hot vector for *text*."""
    v = [0.0] * _DIM
    v[_axis_for(text)] = 1.0
    return v


class _FakeSparseModel:
    """Mimics fastembed SparseTextEmbedding.embed(texts)."""

    class _SE:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    def embed(self, texts):
        for text in texts:
            yield self._SE([_axis_for(text)], [1.0])


class _FakeReranker:
    """Mimics a cohere ClientV2 with .v2.rerank(...). Identity ordering."""

    class _V2:
        def rerank(self, *, model, query, documents, top_n):
            class _R:
                def __init__(self, index, score):
                    self.index = index
                    self.relevance_score = score

            class _Resp:
                def __init__(self, results):
                    self.results = results

            results = [_R(i, 1.0 - i * 0.01) for i in range(min(top_n, len(documents)))]
            return _Resp(results)

    def __init__(self):
        self.v2 = self._V2()


class _FakeLLM:
    """Mimics AnthropicLLM/OpenAILLM .answer(prompt) -> str (plain text)."""

    def answer(self, prompt: str) -> str:
        return "a grounded answer"


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


def _seeded_client() -> QdrantClient:
    """In-memory Qdrant under settings.collection with two dense+sparse chunks."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.collection,
        vectors_config={"dense": models.VectorParams(size=_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    chunks = [_chunk("1-0"), _chunk("1-1")]
    for i, c in enumerate(chunks):
        v = _vec_for(c.chunk_id)
        axis = v.index(1.0)
        client.upsert(
            collection_name=settings.collection,
            points=[
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": v,
                        "sparse": models.SparseVector(indices=[axis], values=[1.0]),
                    },
                    payload=c.model_dump(),
                )
            ],
            wait=True,
        )
    return client


def _pipeline() -> RagPipeline:
    return RagPipeline(
        embed_client=_FakeEmbedClient(),
        sparse_model=_FakeSparseModel(),
        llm=_FakeLLM(),
        qdrant_client=_seeded_client(),
        reranker=_FakeReranker(),
    )


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


def test_retrieve_dense_returns_retrieved_chunks():
    pipe = _pipeline()
    results = pipe.retrieve("text for 1-0", strategy="dense", k=2)
    assert isinstance(results, list)
    assert results
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].chunk.chunk_id == "1-0"


def test_retrieve_hybrid_returns_retrieved_chunks():
    pipe = _pipeline()
    results = pipe.retrieve("text for 1-1", strategy="hybrid", k=2)
    assert results
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_unknown_strategy_raises():
    pipe = _pipeline()
    with pytest.raises(ValueError):
        pipe.retrieve("q", strategy="nope")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


def test_ask_returns_answer_with_deterministic_sources():
    pipe = _pipeline()
    ans = pipe.ask("text for 1-0", strategy="dense", k=2)
    assert isinstance(ans, Answer)
    assert ans.text
    # sources are exactly the retrieved chunks (deterministic, typed) — not the
    # LLM's self-report. The first retrieved is the queried chunk.
    assert ans.sources
    assert ans.sources[0].chunk.chunk_id == "1-0"
    assert all(isinstance(s.chunk.chunk_id, str) for s in ans.sources)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_report_with_one_row_per_strategy():
    pipe = _pipeline()
    golden = [
        GoldenQA(
            question="text for 1-0",
            ground_truth="something",
            relevant_chunk_ids=["1-0"],
        ),
        GoldenQA(
            question="text for 1-1",
            ground_truth="something else",
            relevant_chunk_ids=["1-1"],
        ),
    ]
    strategies = ("dense", "hybrid", "hybrid_rerank")
    report = pipe.evaluate(golden, strategies=strategies)
    assert isinstance(report, EvalReport)
    assert [r.strategy for r in report.rows] == list(strategies)
    # retrieval metrics are populated (floats)
    for row in report.rows:
        assert row.recall_at_k is not None
        assert row.mrr is not None


def test_parse_cache_roundtrip_and_resume_ids(tmp_path):
    """Cached chunks reload in order; cached doc ids drive resume/skip."""
    import json as _json

    from accurag.pipeline import _cached_doc_ids, _load_cached_chunks

    cache = tmp_path / "parse_cache.jsonl"
    rows = [_chunk("1-0", 1), _chunk("1-1", 1), _chunk("4-0", 4)]
    cache.write_text("\n".join(_json.dumps(c.model_dump()) for c in rows) + "\n")

    assert _cached_doc_ids(cache) == {1, 4}
    loaded = _load_cached_chunks(cache)
    assert [c.chunk_id for c in loaded] == ["1-0", "1-1", "4-0"]


def test_cached_doc_ids_missing_file_is_empty(tmp_path):
    from accurag.pipeline import _cached_doc_ids

    assert _cached_doc_ids(tmp_path / "nope.jsonl") == set()


def test_evaluate_matches_doc_level_golden_ids():
    """A doc-level relevant id ('1') must score against retrieved chunk '1-0'.

    Regression for the doc-level vs chunk-level granularity mismatch: without it
    recall/MRR are 0 for every doc-level golden question even when retrieval works.
    """
    pipe = _pipeline()
    golden = [
        GoldenQA(question="text for 1-0", ground_truth="x", relevant_chunk_ids=["1"]),
    ]
    report = pipe.evaluate(golden, strategies=("dense",))
    row = report.rows[0]
    assert row.recall_at_k == 1.0  # doc "1" hit via retrieved chunk "1-0"
    assert row.mrr == 1.0


def test_evaluate_answer_quality_populates_columns():
    """With answer_quality=True + an injected fake judge, faithfulness/answer
    relevancy columns are filled (no real API calls)."""
    import json
    from types import SimpleNamespace

    def _judge_resp():
        tc = SimpleNamespace(
            function=SimpleNamespace(
                arguments=json.dumps({"faithfulness": 1.0, "answer_relevancy": 0.5})
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]))])

    class _FakeJudge:
        chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: _judge_resp()))

    pipe = _pipeline()
    golden = [GoldenQA(question="text for 1-0", ground_truth="x", relevant_chunk_ids=["1"])]
    report = pipe.evaluate(
        golden, strategies=("dense",), answer_quality=True, judge_client=_FakeJudge()
    )
    row = report.rows[0]
    assert row.faithfulness == 1.0
    assert row.answer_relevancy == 0.5


def test_hybrid_rerank_overfetches_candidate_pool(monkeypatch):
    """hybrid_rerank must request a deep candidate pool (settings.rerank_candidates),
    not the shallow k — the reranker needs depth to work (paper [16-27])."""
    import accurag.retrieve as rmod
    from accurag.config import settings

    captured = {}
    real_hybrid = rmod.hybrid

    def spy(client, dense_vec, sparse_vec, k=5, **kw):
        captured["k"] = k
        return real_hybrid(client, dense_vec, sparse_vec, k=2)  # only 2 chunks seeded

    monkeypatch.setattr(rmod, "hybrid", spy)
    pipe = _pipeline()
    pipe.retrieve("text for 1-0", strategy="hybrid_rerank", k=5)
    assert captured["k"] >= settings.rerank_candidates  # over-fetches the configured pool, not k

    # explicit override is honoured
    pipe.retrieve("text for 1-0", strategy="hybrid_rerank", k=5, rerank_candidates=80)
    assert captured["k"] == 80


def test_pipelines_are_instance_configured_and_isolated():
    """Two pipelines, two collections, one client — no global mutation, no leakage."""
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    client = QdrantClient(":memory:")
    for col, cid in [("col_a", "1-1"), ("col_b", "1-2")]:
        client.create_collection(
            col,
            vectors_config={"dense": qm.VectorParams(size=_DIM, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        c = _chunk(cid)
        v = _vec_for(cid)
        client.upsert(
            col,
            points=[
                qm.PointStruct(
                    id=0,
                    vector={
                        "dense": v,
                        "sparse": qm.SparseVector(indices=[v.index(1.0)], values=[1.0]),
                    },
                    payload=c.model_dump(),
                )
            ],
            wait=True,
        )

    rag_a = RagPipeline(
        collection="col_a",
        embed_client=_FakeEmbedClient(),
        sparse_model=_FakeSparseModel(),
        qdrant_client=client,
    )
    rag_b = RagPipeline(
        collection="col_b",
        embed_client=_FakeEmbedClient(),
        sparse_model=_FakeSparseModel(),
        qdrant_client=client,
    )

    assert rag_a.cfg.collection == "col_a"
    assert rag_b.cfg.collection == "col_b"
    # each pipeline queries ITS OWN collection
    assert rag_a.retrieve("text for 1-1", strategy="dense", k=1)[0].chunk.chunk_id == "1-1"
    assert rag_b.retrieve("text for 1-2", strategy="dense", k=1)[0].chunk.chunk_id == "1-2"
    # the global settings was never touched
    from accurag.config import settings as global_settings

    assert global_settings.collection not in ("col_a", "col_b")


def test_embed_model_override_is_used():
    rag = RagPipeline(embed_model="text-embedding-3-small")
    assert rag.cfg.embed_model == "text-embedding-3-small"
    # default instance is unaffected
    assert RagPipeline().cfg.embed_model == "text-embedding-3-large"


def test_per_instance_api_keys_thread_to_clients():
    """Keys passed via config=Settings(...) reach the lazily-built clients —
    not silently overridden by the module global (regression for the leaky
    'no globals' abstraction)."""
    from accurag import Settings
    from accurag.config import settings as global_settings

    rag = RagPipeline(config=Settings(anthropic_api_key="key-A", openai_api_key="key-O"))
    assert rag.llm._api_key == "key-A"  # threaded into AnthropicLLM
    assert rag.embed_client.api_key == "key-O"  # threaded into openai.OpenAI

    # a second pipeline is independent, and the global is untouched
    rag2 = RagPipeline(config=Settings(anthropic_api_key="key-B"))
    assert rag2.llm._api_key == "key-B"
    assert rag.llm._api_key == "key-A"
    assert global_settings.anthropic_api_key not in ("key-A", "key-B")
