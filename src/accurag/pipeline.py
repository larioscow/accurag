"""RagPipeline — the facade that wires every accurag module into one object.

The pipeline composes the corpus-agnostic building blocks (fetch, parse, chunk,
embed, sparse_embed, index, retrieve, rerank, answer, evaluate) behind four
methods: :meth:`ingest`, :meth:`retrieve`, :meth:`ask`, :meth:`evaluate`.

Design rules (mirrors the rest of the library):
- Every heavy / key-requiring collaborator is *injectable* via the constructor
  and only built lazily on first real use, so importing this module — and
  constructing a pipeline with fakes — needs no API keys and no network.
- The default LLM is :class:`~accurag.llm.AnthropicLLM` (Claude); swap in
  :class:`~accurag.llm.OpenAILLM` for the documented OpenAI fallback.
- The expensive ingestion stages (parse + chunk + embed) are cached to
  ``settings.chunks_path`` as JSONL so re-indexing is cheap.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from accurag.config import Settings, settings
from accurag.models import (
    Answer,
    Chunk,
    EvalReport,
    EvalRow,
    GoldenQA,
    ManifestEntry,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

_STRATEGIES = ("dense", "hybrid", "hybrid_rerank")


class RagPipeline:
    """End-to-end RAG pipeline over a fixed corpus + Qdrant index.

    Configuration is per-instance — no global state. ``RagPipeline()`` uses the
    package defaults; pass ``collection=`` / ``embed_model=`` (or a full
    ``config=Settings(...)``) to run multiple independent pipelines in one process
    without touching any global:

        rag  = RagPipeline()                                  # defaults
        docs = RagPipeline(collection="my_docs")              # different corpus
        fast = RagPipeline(embed_model="text-embedding-3-small")

    Args:
        embed_client:   OpenAI-compatible embeddings client (``embed.make_client``).
        sparse_model:   fastembed sparse model (``sparse_embed.make_sparse_model``).
        llm:            Object with ``.answer(prompt) -> str``; defaults to
                        :class:`~accurag.llm.AnthropicLLM` (OpenAI fallback noted).
        qdrant_client:  A ``QdrantClient`` (``index.make_client`` by default).
        reranker:       A Cohere-compatible client for ``rerank.rerank``.
        config:         A :class:`~accurag.config.Settings` to base config on
                        (defaults to the package settings; a copy is taken so the
                        instance is isolated from later global mutation).
        collection:     convenience override for the corpus/index name (cfg.collection).
        embed_model:    convenience override for the dense embedder (cfg.embed_model).

    Note on keys: ``config``'s API keys are applied only to the *default*,
    lazily-built clients. If you inject your own ``embed_client`` / ``llm`` /
    ``reranker``, that client owns its own credentials — the matching
    ``config`` key is not applied to it (you already configured it).
    """

    def __init__(
        self,
        embed_client: Any = None,
        sparse_model: Any = None,
        llm: Any = None,
        qdrant_client: Any = None,
        reranker: Any = None,
        config: Settings | None = None,
        collection: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._embed_client = embed_client
        self._sparse_model = sparse_model
        self._llm = llm
        self._qdrant = qdrant_client
        self._reranker = reranker

        overrides = {
            k: v
            for k, v in {"collection": collection, "embed_model": embed_model}.items()
            if v is not None
        }
        # Copy so this instance is isolated (mutating the global later, or another
        # pipeline's config, never affects this one).
        self.cfg: Settings = (config if config is not None else settings).model_copy(
            update=overrides
        )

    # ------------------------------------------------------------------
    # Lazy collaborator accessors
    # ------------------------------------------------------------------

    @property
    def embed_client(self) -> Any:
        if self._embed_client is None:
            from accurag import embed

            self._embed_client = embed.make_client(api_key=self.cfg.openai_api_key)
        return self._embed_client

    @property
    def sparse_model(self) -> Any:
        if self._sparse_model is None:
            from accurag import sparse_embed

            self._sparse_model = sparse_embed.make_sparse_model()
        return self._sparse_model

    @property
    def llm(self) -> Any:
        if self._llm is None:
            # Default: Anthropic Claude. OpenAILLM is the documented fallback.
            from accurag.llm import AnthropicLLM

            self._llm = AnthropicLLM(api_key=self.cfg.anthropic_api_key)
        return self._llm

    @property
    def reranker(self) -> Any:
        """The Cohere-compatible rerank client (built lazily from cfg key)."""
        if self._reranker is None:
            from accurag import rerank

            self._reranker = rerank._make_client(api_key=self.cfg.cohere_api_key)
        return self._reranker

    @property
    def qdrant(self) -> Any:
        if self._qdrant is None:
            from accurag import index

            self._qdrant = index.make_client(path=self.cfg.qdrant_path)
        return self._qdrant

    # ------------------------------------------------------------------
    # Ingestion (offline)
    # ------------------------------------------------------------------

    def ingest(
        self,
        manifest_path: Path | None = None,
        limit: int | None = None,
        parser: str = "docling",
        chunk_size: int | None = None,
        vision: bool = False,
    ) -> int:
        """Fetch -> parse -> chunk -> embed -> index the whole corpus.

        Per-document parse/chunk is wrapped in try/except so one bad document
        cannot abort the run. The resulting chunks + their dense/sparse vectors
        are persisted to ``settings.chunks_path`` (JSONL) as the expensive-stage
        cache before being upserted into Qdrant.

        Args:
            manifest_path: Path to the corpus manifest JSON.
            limit:         Optional cap on number of manifest entries to ingest.

        Returns:
            The number of chunks indexed.
        """
        from accurag import chunk as chunk_mod
        from accurag import embed, fetch, index, sparse_embed

        manifest_path = manifest_path if manifest_path is not None else self.cfg.manifest_path
        entries: list[ManifestEntry] = fetch.load_manifest(manifest_path)
        if limit is not None:
            entries = entries[:limit]

        by_id = {e.id: e for e in entries}
        paths = fetch.fetch_all(entries, self.cfg.raw_dir)

        # Chunk-size budget = min(target, embedder ceiling). The embedder window
        # is a hard ceiling (exceed it -> silent truncation); the target is the
        # quality knob. min() keeps chunks within whatever embedder we're using.
        target = chunk_size or self.cfg.chunk_size
        ceiling = embed.embedder_max_tokens(self.embed_client)
        budget = min(target, ceiling)
        logger.info(
            "[ingest] chunk budget=%d tokens (target=%d, embedder ceiling=%d)",
            budget,
            target,
            ceiling,
        )

        # CRASH-SAFE PARSE: each doc's chunks are appended to a cache file the
        # instant they're parsed. A crash/kill (or a segfault in the later embed
        # step) never loses parse work — re-running ingest skips docs already in
        # the cache and resumes. The expensive CPU parse therefore happens once.
        cache_path = self.cfg.parse_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        done_doc_ids = _cached_doc_ids(cache_path)
        if done_doc_ids:
            logger.info("[ingest] resuming — %d docs already in parse cache", len(done_doc_ids))

        total = len(paths)
        with cache_path.open("a", encoding="utf-8") as cache:
            for i, path in enumerate(paths, start=1):
                doc_id = _doc_id_from_path(path)
                entry = by_id.get(doc_id) if doc_id is not None else None
                if entry is None:
                    logger.warning("no manifest entry for %s — skipping", path)
                    continue
                if entry.id in done_doc_ids:
                    logger.info("[ingest] %d/%d  doc %s  cached — skip parse", i, total, entry.id)
                    continue
                try:
                    if parser == "pymupdf":
                        from accurag import fastparse

                        text = fastparse.extract_text(path)
                        doc_chunks = chunk_mod.chunk_text(text, entry, max_tokens=budget)
                    else:
                        from accurag import parse

                        doc = parse.parse_file(path)
                        doc_chunks = chunk_mod.chunk_document(doc, entry, max_tokens=budget)
                    if vision and path.suffix.lower() == ".pdf":
                        # Optional: read figures/charts into text chunks (extra
                        # Docling parse + a vision model call per figure). Isolated
                        # in its own try so a vision failure never costs us the
                        # already-parsed TEXT chunks — figures are best-effort.
                        try:
                            from accurag import vision as vision_mod

                            figs = vision_mod.figure_chunks(
                                path, entry, model=self.cfg.vision_model
                            )
                            if figs:
                                logger.info(
                                    "[ingest] doc %s: +%d figure chunks", entry.id, len(figs)
                                )
                                doc_chunks = doc_chunks + figs
                        except Exception as vexc:  # noqa: BLE001 — figures are optional
                            logger.warning(
                                "[ingest] doc %s: figure extraction failed (%s) — "
                                "keeping text chunks",
                                entry.id,
                                vexc,
                            )
                    for c in doc_chunks:
                        cache.write(json.dumps(c.model_dump()) + "\n")
                    cache.flush()  # persist this doc before moving on
                    done_doc_ids.add(entry.id)
                    logger.info(
                        "[ingest] parsed %d/%d  doc %s  +%d chunks (cached)",
                        i,
                        total,
                        entry.id,
                        len(doc_chunks),
                    )
                except Exception as exc:  # noqa: BLE001 — one bad doc must not kill the run
                    logger.warning("[ingest] failed doc %s (%s): %s", entry.id, path.name, exc)

        all_chunks = _load_cached_chunks(cache_path)
        if not all_chunks:
            logger.warning("ingest produced no chunks")
            return 0

        logger.info("[ingest] parse done: %d chunks — embedding (dense+sparse)…", len(all_chunks))
        texts = [c.text for c in all_chunks]
        dense_vectors = embed.embed_texts(texts, self.embed_client, model=self.cfg.embed_model)
        sparse_vectors = sparse_embed.sparse_embed_texts(texts, self.sparse_model)

        self._persist_chunks(all_chunks, dense_vectors, sparse_vectors)

        logger.info("[ingest] embedded — building collection + indexing into Qdrant…")
        client = self.qdrant
        index.build_collection(
            client, dim=len(dense_vectors[0]), with_sparse=True, collection=self.cfg.collection
        )
        index.index_chunks(
            client, all_chunks, dense_vectors, sparse_vectors, collection=self.cfg.collection
        )

        logger.info("ingested %d chunks", len(all_chunks))
        return len(all_chunks)

    def _persist_chunks(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> None:
        """Write chunks + vectors to the configured chunks_path as JSONL (the cache)."""
        path = self.cfg.chunks_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
                indices, values = sparse
                record = {
                    "chunk": chunk.model_dump(),
                    "dense": dense,
                    "sparse": {"indices": indices, "values": values},
                }
                fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        strategy: str = "dense",
        k: int = 5,
        rerank_candidates: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-*k* chunks for *query* under *strategy*.

        Args:
            query:    The user's question.
            strategy: One of ``"dense"``, ``"hybrid"``, ``"hybrid_rerank"``.
            k:        Number of chunks to return.
            rerank_candidates: How many candidates to over-fetch before reranking
                      (``hybrid_rerank`` only). Defaults to ``settings.rerank_candidates``
                      (20). A deeper pool (50) was measured and did slightly worse
                      on the bundled corpus (first-stage recall is saturated — see
                      docs/EVAL_RESULTS.md); tune it up for a harder corpus.

        Returns:
            Ranked list of :class:`RetrievedChunk` (rank 0 = best).
        """
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {_STRATEGIES}")

        from accurag import embed, rerank, retrieve, sparse_embed

        col = self.cfg.collection
        dense_vec = embed.embed_texts([query], self.embed_client, model=self.cfg.embed_model)[0]

        if strategy == "dense":
            return retrieve.dense(self.qdrant, dense_vec, k=k, collection=col)

        # hybrid + hybrid_rerank both need a sparse query vector
        sparse_vec = sparse_embed.sparse_embed_texts([query], self.sparse_model)[0]

        if strategy == "hybrid":
            return retrieve.hybrid(self.qdrant, dense_vec, sparse_vec, k=k, collection=col)

        # hybrid_rerank: over-fetch a deep candidate pool, then rerank down to k.
        candidates = rerank_candidates or self.cfg.rerank_candidates
        fetched = retrieve.hybrid(
            self.qdrant, dense_vec, sparse_vec, k=max(k, candidates), collection=col
        )
        if not fetched:
            return []  # nothing to rerank — skip building the Cohere client
        return rerank.rerank(query, fetched, top_n=k, client=self.reranker)

    # ------------------------------------------------------------------
    # Ask (retrieve + grounded answer)
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        strategy: str = "dense",
        k: int = 5,
    ) -> Answer:
        """Retrieve context for *query* and return a grounded Answer.

        ``Answer.sources`` is exactly the set of chunks that were retrieved and
        fed to the model — deterministic provenance set in Python, not
        self-reported by the LLM. There is no hallucinated citation to validate:
        every source is, by construction, a chunk the answer was grounded on.
        """
        from accurag.answer import build_answer

        retrieved = self.retrieve(query, strategy=strategy, k=k)
        return build_answer(query, retrieved, self.llm)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        golden: list[GoldenQA],
        strategies: tuple[str, ...] = _STRATEGIES,
        k: int = 5,
        answer_quality: bool = False,
        judge_client: Any = None,
    ) -> EvalReport:
        """Score each *strategy* over the *golden* set and build the report.

        Always computes retrieval metrics (recall@k, MRR) and p95 latency per
        strategy. When ``answer_quality`` is True, also generates an answer per
        question (via the configured generator LLM) and grades faithfulness +
        answer relevancy with a separate judge LLM (gpt-4o-mini ≠ generator) —
        see :func:`accurag.evaluate.judge_answer_quality`. That path makes
        generator + judge API calls, so it is opt-in.

        Returns:
            An :class:`EvalReport` with one :class:`EvalRow` per strategy.
        """
        from accurag import evaluate as ev
        from accurag.answer import build_answer
        from accurag.llm import EmptyCompletionError

        judge = judge_client or (ev.make_judge_client() if answer_quality else None)

        rows: list[EvalRow] = []
        for strategy in strategies:
            recalls: list[float] = []
            rrs: list[float] = []
            latencies: list[float] = []
            samples: list[dict[str, Any]] = []

            for qa in golden:
                start = time.perf_counter()
                retrieved = self.retrieve(qa.question, strategy=strategy, k=k)
                latencies.append((time.perf_counter() - start) * 1000.0)

                # The golden set may label relevance at DOC level (manifest id,
                # e.g. "12") or CHUNK level ("12-3"). Compare at matching
                # granularity so doc-level placeholders score meaningfully — a
                # retrieved chunk "12-3" counts as a hit for relevant doc "12".
                relevant = qa.relevant_chunk_ids
                doc_level = bool(relevant) and all("-" not in str(r) for r in relevant)
                if doc_level:
                    ranked_ids = [str(rc.chunk.doc_id) for rc in retrieved]
                else:
                    ranked_ids = [rc.chunk.chunk_id for rc in retrieved]
                recalls.append(ev.recall_at_k(relevant, ranked_ids))
                rrs.append(ev.mrr(relevant, ranked_ids))

                if answer_quality:
                    # ask() raises on an empty/refused completion (so a single
                    # call surfaces the failure loudly). In a batch eval, though,
                    # one empty answer must not abort the run — record it empty
                    # (the judge scores it unfaithful) and move on. Caught
                    # narrowly: a real API/network error still propagates.
                    try:
                        answer_text = build_answer(qa.question, retrieved, self.llm).text
                    except EmptyCompletionError as exc:
                        logger.warning("[eval] no answer for %r (%s)", qa.question, exc)
                        answer_text = ""
                    samples.append(
                        {
                            "question": qa.question,
                            "answer": answer_text,
                            "contexts": [rc.chunk.text for rc in retrieved],
                        }
                    )

            quality = ev.judge_answer_quality(samples, client=judge) if answer_quality else {}
            rows.append(
                EvalRow(
                    strategy=strategy,
                    recall_at_k=_mean(recalls),
                    mrr=_mean(rrs),
                    faithfulness=quality.get("faithfulness"),
                    answer_relevancy=quality.get("answer_relevancy"),
                    judge_coverage=quality.get("coverage"),
                    p95_latency_ms=_p95(latencies),
                )
            )

        return ev.build_report(rows)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _doc_id_from_path(path: Path) -> int | None:
    """Recover the manifest doc id from a fetched file path (``<id>.pdf``)."""
    try:
        return int(Path(path).stem)
    except ValueError:
        return None


def _cached_doc_ids(cache_path: Path) -> set[int]:
    """Doc ids already present in the parse cache (for resume/skip)."""
    ids: set[int] = set()
    if not cache_path.exists():
        return ids
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(json.loads(line)["doc_id"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return ids


def _load_cached_chunks(cache_path: Path) -> list[Chunk]:
    """Load all parsed chunks from the parse cache, in file order."""
    chunks: list[Chunk] = []
    if not cache_path.exists():
        return chunks
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> float | None:
    """95th-percentile (nearest-rank) of *values* in ms."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, round(0.95 * (len(ordered) - 1)))
    return ordered[idx]
