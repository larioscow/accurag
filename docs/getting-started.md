# Getting started

accurag is a lean, evaluation-driven RAG **library** — not a deployed app, not a
framework. You `import accurag` (or drive the CLI) and get four verbs — **ingest /
retrieve / ask / evaluate** — over one strategy switch: `dense | hybrid |
hybrid_rerank`. This guide takes you from install to a grounded answer, top to bottom.

## Install

accurag is pip-installable. The base install carries 8 declared core packages (lazy
imports keep `import accurag` cheap and key-free); the heavy parsing, eval, and
observability SDKs sit behind optional extras you opt into.

```bash
git clone <this repo> && cd accurag
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingest]"     # parsers (Docling + PyMuPDF + tiktoken) live behind this extra
```

The available extras (from `pyproject.toml`):

| Extra | Pulls in | When you need it |
|---|---|---|
| `ingest` | `docling`, `pymupdf`, `tiktoken` | to parse + chunk source documents during ingestion |
| `eval` | `ragas` | only for the optional Ragas eval path (the default judge is in-library and needs no extra) |
| `obs` | `langfuse` | optional observability |
| `dev` | `pytest`, `ruff`, `mypy` | running the test suite + linters |
| `docs` | `mkdocs-material`, `mkdocstrings`, … | building this documentation site |

Combine extras as needed, e.g. `pip install -e ".[ingest,dev]"`.

!!! note
    `langchain` is never *imported* by accurag. It can arrive transitively if you
    install the `eval` extra, which brings Ragas.

## Keys

| Variable | Used for | Required? |
|---|---|---|
| `OPENAI_API_KEY` | embeddings + the eval judge | no — `--embedder local` avoids it |
| `ANTHROPIC_API_KEY` | answer generation (Claude) | for `ask`; or pass `--llm openai` |
| `COHERE_API_KEY` | reranking | only for `hybrid_rerank` (a free trial key works) |

```bash
cp .env.example .env     # then add the keys you need
```

If you want to try ingest + retrieve with **no key at all**, skip ahead to
[No-API mode](#no-api-offline-mode).

## The four verbs, from Python

The whole public surface is one object, `RagPipeline`, and four methods.

```python
from accurag import RagPipeline

rag = RagPipeline()

# 1. ingest — parse → chunk → embed → index a corpus into Qdrant (offline, slow)
rag.ingest("corpus/manifest.json")

# 2. retrieve — ranked chunks, no LLM call
hits = rag.retrieve("How does reciprocal rank fusion work?", strategy="hybrid")
for h in hits:
    print(h.rank, round(h.score, 3), h.chunk.chunk_id, h.chunk.source_title)

# 3. ask — a grounded Answer plus the sources it was conditioned on
ans = rag.ask("How does reciprocal rank fusion work?", strategy="hybrid_rerank")
print(ans.text)                        # plain prose — no inline citation markers
for s in ans.sources:                  # typed RetrievedChunk provenance
    print(s.chunk.source_title, s.chunk.url, s.score)
```

`ask` returns an `Answer` with two fields:

- `Answer.text` — the generated answer as plain prose.
- `Answer.sources` — a `list[RetrievedChunk]`, each with `.chunk`, `.score`, and
  `.rank`.

`Answer.sources` is the set of chunks that were retrieved and fed to the model —
**assigned in Python, not self-reported by the LLM**. This eliminates the
*hallucinated / nonexistent-citation* class entirely: there is no model-reported chunk
id to validate, because the prompt forbids the model from emitting citations at all.

!!! warning "What deterministic provenance does NOT buy"
    A source is guaranteed real and guaranteed to be the one used — *not* guaranteed
    to have been read correctly. A wrong answer drawn from a real-but-bad chunk is not
    caught. There is a documented vision limit: chart *structure* (titles, axes,
    series) reads reliably, but bar/line *heights are guessed*, so a quantitative chart
    answer can come back grounded but numerically wrong. Provenance is not correctness.

### The strategy switch

`retrieve` and `ask` both take `strategy=`, the one knob the whole project is built
around — it *is* the comparison the eval measures:

- `dense` — vector search only.
- `hybrid` — dense + sparse (BM25-style), fused with reciprocal rank fusion.
- `hybrid_rerank` — hybrid, then a Cohere reranker over an over-fetched candidate pool
  (needs `COHERE_API_KEY`).

```python
rag.retrieve("question", strategy="dense")
rag.retrieve("question", strategy="hybrid")
rag.ask("question", strategy="hybrid_rerank")
```

### evaluate

The fourth verb scores each strategy over a hand-built golden set and builds the
comparison report. Retrieval metrics (recall@k, MRR, p95 latency) are always computed;
answer-quality columns (faithfulness, answer relevancy) are opt-in via
`answer_quality=True` and are graded by an in-library LLM judge that is a **different
model** from the generator.

```python
from accurag import GoldenQA

golden = [
    GoldenQA(
        question="How does reciprocal rank fusion work?",
        ground_truth="RRF combines rankings by summing 1/(k + rank) across lists.",
        relevant_chunk_ids=["12"],   # doc-level id, or "12-3" for a chunk-level label
    ),
    # … more examples
]

report = rag.evaluate(golden, strategies=["dense", "hybrid", "hybrid_rerank"])
print(report.to_markdown())          # the comparison table
```

!!! note
    The in-library judge wraps each sample in its own try/except and reports a
    **coverage** field, so one bad judge reply skips that sample instead of aborting
    the run. This robustness is on the **default in-library path only** — the optional
    Ragas path and malformed *inputs* are not guarded.

## The same four verbs, from the CLI

```bash
python -m accurag.cli ingest --parser docling                       # ~15 min CPU, ~$0.10 embeddings
python -m accurag.cli ask "How does reciprocal rank fusion work?" --strategy hybrid
python -m accurag.cli evaluate data/golden/golden.jsonl --answer-quality
```

## No-API (offline) mode

You don't need paid embeddings to try it. `--embedder local` swaps OpenAI for a local
model (fastembed, CPU-only), so **ingest + retrieve run with no key and no network**
(after the model is cached on first use):

```bash
python -m accurag.cli ingest --parser pymupdf --embedder local
python -m accurag.cli evaluate data/golden/golden.jsonl --embedder local --strategies dense,hybrid
```

The Python equivalent injects `LocalEmbeddingClient` (default model
`BAAI/bge-small-en-v1.5`, 384-dim):

```python
from accurag import RagPipeline, LocalEmbeddingClient

rag = RagPipeline(embed_client=LocalEmbeddingClient())
rag.ingest(limit=2)                                  # fully offline
rag.retrieve("What is retrieval-augmented generation?", strategy="hybrid", k=3)
```

A runnable end-to-end version of this lives at `examples/local_offline_smoke.py`.

!!! warning ".ask still needs a hosted LLM"
    `--embedder local` covers **ingest + retrieval only** — it is "offline for ingest +
    retrieval," not fully offline. Answering (`.ask`) still needs a hosted LLM:
    generation is the one part `--embedder local` can't cover. Set `ANTHROPIC_API_KEY`
    (or pass `--llm openai` with an OpenAI key) when you want grounded answers.

!!! note "Keep the embedder consistent"
    The local embedding dimension differs from OpenAI's, so an index built with
    `LocalEmbeddingClient` must also be *queried* with it. Don't mix embedders across
    ingest and retrieve.

## Where to go next

- The measured comparison table and the negative results that drove the design:
  [EVAL_RESULTS.md](EVAL_RESULTS.md).
- The full API surface — per-instance config, swapping any piece (LLM, reranker,
  collection): the [API reference](reference/accurag/index.md).
</content>
