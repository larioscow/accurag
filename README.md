# accurag

[![CI](https://github.com/larioscow/accurag/actions/workflows/ci.yml/badge.svg)](https://github.com/larioscow/accurag/actions/workflows/ci.yml)
[![docs](https://github.com/larioscow/accurag/actions/workflows/docs.yml/badge.svg)](https://larioscow.github.io/accurag/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

**[Documentation site](https://larioscow.github.io/accurag/)** — guides, architecture, and an auto-generated API reference.

A lean, evaluation-driven RAG **library** — not a deployed app, not a framework. You
`import accurag` (or drive the CLI) and get four verbs — **ingest / retrieve / ask /
evaluate** — over one strategy switch: `dense | hybrid | hybrid_rerank`. That switch
*is* the comparison the eval measures.

The reason it exists is the evaluation. Which parser, which retrieval strategy, what
chunk size — every one of those was decided by running the eval, not by argument. And
the sources an answer returns are **the retrieved chunks themselves, assigned in
Python** — provenance is deterministic, never self-reported by the model.

**Headline (full numbers below, full breakdown in
[`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md)):** on the shipped corpus, hybrid
retrieval reaches **recall@5 0.947** (doc-level — chunk-level is ~0.53, explained
below) and faithfulness sits at **~0.98** across all three strategies. The most useful
result is a *negative* one: deepening the reranker's candidate pool 20 → 50 was measured
and **slightly hurt** recall — so it wasn't shipped. That section is the point of the
project.

The corpus it ships with is **~47 clean AI/RAG technical papers and vendor docs** (the
original RAG paper, DPR, ColBERT, Lost-in-the-Middle, Self-RAG, Anthropic's
contextual-retrieval appendix, and so on). Be upfront about it: this is an **easy**
corpus — clean academic PDFs — and first-stage retrieval recall is **saturated** on it.
That is not a flattering accident; it is *why* the retrieval upgrades below show little
headroom here, and the README says so rather than hiding it.

---

## Architecture

Two halves joined only by the Qdrant index. Ingestion is slow, heavy, and offline (it
runs a parser); serving is fast and never touches the parser — the parsing SDKs live
behind an optional extra so a serving install doesn't carry them.

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion — offline, once (extra: [ingest])"]
        direction LR
        SRC[manifest.json<br/>source docs] --> PARSE[parse<br/>Docling / PyMuPDF]
        PARSE --> CHUNK[structure-aware chunk<br/>+ metadata]
        CHUNK --> EMB1[dense embed<br/>OpenAI / local fastembed]
        CHUNK --> EMB2[sparse embed<br/>BM25-style]
    end

    EMB1 --> QDRANT[(Qdrant<br/>embedded, in-process)]
    EMB2 --> QDRANT

    subgraph Serving["Serving — per query"]
        direction LR
        Q[question] --> QEMB[embed query]
        QEMB --> RET[hybrid retrieve<br/>dense + sparse, RRF]
        RET --> RERANK[rerank<br/>Cohere, optional]
        RERANK --> GEN[grounded answer<br/>Claude / OpenAI — plain prose]
    end

    QDRANT --> RET
    GEN --> ANS[Answer.text]
    RET -. retrieved chunks .-> SRCS[Answer.sources<br/>typed RetrievedChunk<br/>assigned in Python]

    classDef store fill:#1f2937,stroke:#94a3b8,color:#e5e7eb;
    class QDRANT store;
```

The dotted edge is the load-bearing design choice: `Answer.sources` is wired straight
from the **retrieved chunks**, *not* from anything the generator emits. More on why
below.

Internals are deliberately legible: **~19 small modules**, every stage a plain typed
function, and strategy selection is a literal `if/elif` over three named functions — no
registry, no dynamic dispatch. `pipeline.py` is the only file that wires them together.
Retrieval returns `RetrievedChunk{chunk, score, rank}`.

---

## Design decisions, and why

### Deterministic sources over LLM-reported citations

The generator writes **plain prose**. The prompt explicitly *forbids* it from emitting
citations. `Answer.sources` is set in Python to the exact `RetrievedChunk` objects that
were fed to the model.

**What this buys (qualified, because honesty is the point):** it eliminates the
*hallucinated / nonexistent-citation* class entirely — there is no model-reported chunk
id to validate, because the model reports none. A consumer (e.g. a NotebookLM-style UI)
gets sources as typed data, never as a string to parse out of the answer.

**What it does NOT buy:** it does not catch a *wrong answer drawn from a real-but-bad
chunk*. The source is guaranteed real and guaranteed to be the one used — not guaranteed
to have been read correctly. There is a documented failure mode where vision-extracted
chart numbers are mis-read and the answer comes back **grounded but numerically wrong**,
faithful to a mis-extracted source. Deterministic provenance does not save you there,
and the eval doc says so plainly.

### Hybrid over pure vector, then rerank on top

Vector search alone misses exact terms and numbers; sparse/lexical matching catches
them — that is the recall lift in the table (0.867 → 0.947 doc-level). Reranking then
floats the best chunk to rank 1 (MRR 0.828 → 0.861) at a small recall cost, because it
truncates the over-fetched set. Not a clean sweep — a real, measured tradeoff.

### A library, not a framework — and no LangChain

The public surface is four verbs and one switch. No HTTP server, UI, auth, or deploy
lives here; those belong to a separate app that would `import` this. The eval is built
**into** the library rather than bolted on with Ragas: a dependency-free LLM judge,
~80 lines over the OpenAI SDK already in use. Ragas pulls the LangChain dependency tree
and, in the version hit, broke on a missing `langchain_community` import. The judge is
also a **different model** from the generator (`gpt-4o-mini` judging Claude's answers),
so grader and graded aren't the same system.

A note on dependencies, stated precisely: **8 declared core packages**; heavy
parsing/eval/observability SDKs sit behind optional extras with lazy in-function
imports, so `import accurag` is key-free and cheap. Qualifier: 5 of those 8 are real
SDKs; and `langchain` is never *imported* by accurag, though it can arrive transitively
if you install the optional eval extra (which brings Ragas).

### Robust eval judge (the failure mode Ragas has)

The in-library judge wraps each sample in its own try/except and reports a **coverage**
field, so one bad judge reply skips that sample instead of aborting the whole run.
Qualifier: this robustness is on the **default in-library path**. The optional Ragas
path and malformed *inputs* are not guarded.

---

## The result it's built around

Three retrieval strategies, scored on a hand-built golden set of **25 Q&A** over all 47
documents (Docling parser, 4,215 chunks). Faithfulness and answer relevancy are graded
by the in-library LLM judge (`gpt-4o-mini`), which is **not** the model writing the
answers (Claude).

| Config | Faithfulness | Answer Rel. | Recall@5 | MRR | P95 Latency (ms)* |
|---|---|---|---|---|---|
| dense (naive) | 0.988 | 0.776 | 0.867 | 0.803 | 670 |
| hybrid (dense+sparse, RRF) | 0.978 | 0.776 | **0.947** | 0.828 | 454 |
| hybrid + rerank (Cohere) | 0.984 | **0.808** | 0.933 | **0.861** | 1208 |

\* Rerank's 1208 ms is inflated by a Cohere **trial key** (10 calls/min, backoff under
load), not the reranker itself; treat dense/hybrid (~0.5 s) as representative and rerank
as an upper bound. Latency is indicative, not benchmarked.

What it says, honestly:

- **Faithfulness ~0.98 across the board** — the grounded prompt (answer only from the
  numbered context, else "I don't know") genuinely prevents hallucination; answers stay
  tied to the retrieved context that is returned verbatim as the sources.
- **Hybrid wins recall@5 (0.947)** by adding lexical matching; **rerank wins MRR
  (0.861)** by floating the right chunk to rank 1, at a small recall cost.
- **Answer relevancy ~0.78** — moderate, and honestly so: dragged down by questions the
  corpus can't fully answer, where the system refuses ("I don't know") rather than
  inventing something, and a refusal scores ~0 on relevancy by design. That refusal is
  the behaviour you want.

**Important qualifier on recall.** The 0.947 is **doc-level** recall. With stricter
**chunk-level** labels (the exact answer-bearing chunks), recall@5 drops to ~0.53 — not
because retrieval degraded, but because there are ~4.9 relevant chunks/question against
only k=5 slots, so recall@5 is mathematically capped (|relevant| > k), and tighter
labels expose that hybrid's doc-level lead was partly "retrieved *some* chunk from the
right doc." MRR holds (~0.72–0.74). Both label sets exist and each is used where it's
valid; the full breakdown is in [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md).

### A measured non-result (the senior bit)

A paper *in this corpus* claims feeding a reranker only 20 candidates is too shallow, so
the obvious move was to deepen the pool 20 → 50. **Measured: it slightly HURT** recall@5
(0.933 → 0.893). The mechanism is quantified — hybrid first-stage recall is **flat at
0.947 from depth 5 to 50**, so every relevant doc the first stage will ever find is
already in the top 5; a deeper pool just feeds the reranker distractors. (That 0.947 is
doc-level; chunk-level is ~0.53.) The lesson isn't the 0.04 — it's that a plausible
improvement was *measured and rejected with a mechanism*, not shipped on faith.

Contextual retrieval and late chunking were **not measured**. They were predicted to
have no headroom from the same saturation and parked behind an explicit harder-corpus
test — so "measured" is claimed only for rerank-depth, nowhere else.

The same harness also produced a **parser A/B** (Docling beats fast PyMuPDF extraction
marginally on aggregate retrieval, and ranks table-heavy docs higher — though on the one
table-comparison question both parsers still correctly refused, so the ranking win
didn't convert to an answer) and a **chunk-size sweep** (256 / 512 / 800 tokens — 512
wins, which is why it's the default). Both are in
[`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md).

---

## Quick start

```bash
git clone <this repo> && cd accurag
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingest]"          # parsers (Docling + PyMuPDF) live behind this extra
cp .env.example .env                # add your keys (see "Keys")
```

The four verbs and the strategy switch, from Python:

```python
from accurag import RagPipeline

rag = RagPipeline()
rag.ingest("corpus/manifest.json")                    # parse → chunk → embed → index
hits = rag.retrieve("question", strategy="hybrid")     # ranked chunks, no LLM
ans  = rag.ask("question", strategy="hybrid_rerank")   # grounded Answer + sources
print(ans.text)                                        # clean prose — no inline markers
for s in ans.sources:                                  # typed RetrievedChunk provenance
    print(s.chunk.source_title, s.chunk.url, s.score)
report = rag.evaluate(golden, strategies=["dense", "hybrid", "hybrid_rerank"])
print(report.to_markdown())                            # the table above
```

Or from the CLI:

```bash
python -m accurag.cli ingest --parser docling                       # ~15 min CPU, ~$0.10 embeddings
python -m accurag.cli ask "How does reciprocal rank fusion work?" --strategy hybrid
python -m accurag.cli evaluate data/golden/golden.jsonl --answer-quality
```

### No-API mode

You don't need paid embeddings to try it. `--embedder local` swaps OpenAI for a local
model (fastembed, CPU-only), so **ingest + retrieve run with no key and no network**:

```bash
python -m accurag.cli ingest --parser pymupdf --embedder local
python -m accurag.cli evaluate data/golden/golden.jsonl --embedder local --strategies dense,hybrid
```

Qualifier: this is "offline for ingest + retrieval," **not** fully offline. Answering
(`.ask`) still needs a hosted LLM — generation is the one part `--embedder local` can't
cover.

### Keys

| Variable | Used for | Required? |
|---|---|---|
| `OPENAI_API_KEY` | embeddings + the eval judge | no — `--embedder local` avoids it |
| `ANTHROPIC_API_KEY` | answer generation (Claude) | for `ask`; or pass `--llm openai` |
| `COHERE_API_KEY` | reranking | only for `hybrid_rerank` (a free trial key works) |

Everything is per-instance, no globals — you can run several independent pipelines in one
process:

```python
docs = RagPipeline(collection="my_docs")                  # different corpus
fast = RagPipeline(embed_model="text-embedding-3-small")  # cheaper embeddings
rag  = RagPipeline(llm=OpenAILLM(), reranker=my_cohere_client)  # swap any piece
```

---

## Right-sized by design

pip-installable, **Qdrant embedded in-process** (no separate service for dev),
**CPU-only — there is no GPU path anywhere**. Indexing the 47-doc corpus costs about
**$0.10** of OpenAI embeddings (or $0 with `--embedder local`); a query is a few tenths
of a cent; the full eval with answer-quality columns is well under a dollar.

---

## Limitations, honestly

- **It's a library.** No deployed URL, dashboard, auth, or rate limiting — those belong
  to a separate app that would `import` this. Not built here.
- **The corpus is clean academic PDFs, which makes retrieval easy.** First-stage recall
  is saturated (0.947, flat with depth), so retrieval-improving techniques show no
  headroom *here*. They may well help on messier data.
- **Vision is numerically unreliable — a documented limitation, not a feature.** With
  the cheap vision model, chart *structure* (titles, axes, series) reads reliably but
  *bar/line heights are guessed*, so a quantitative chart answer comes back grounded but
  numerically wrong. This is exactly the failure deterministic provenance does **not**
  catch.
- **Context Precision / Recall (LLM-judged) columns are still `—`** — deterministic
  recall@5 / MRR are reported instead; the judged context columns are a future add.

## What I'd do next

- A **harder, messier corpus** to give the parked retrieval upgrades (deeper reranking,
  contextual retrieval, HyDE) a fair test where first-stage recall is *not* saturated —
  see [`docs/POSSIBLE_UPGRADES.md`](docs/POSSIBLE_UPGRADES.md).
- **Per-claim attribution** done post-hoc and *verifiably* — mapping a sentence to its
  supporting chunk — never by trusting the generator's word.
- A two-tier **vision** path (cheap model for structure, a stronger model routed to
  dense/quantitative charts) to chip at the numerically-wrong failure mode.
- The **deployed showcase app** (FastAPI + Streamlit + auth + rate-limit + dashboard)
  that imports this library. Separate project.
- Deliberately **out of scope** (know when to use, don't build): agentic / query
  routing, knowledge graphs, late chunking, fine-tuned embeddings.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q          # 140 tests, no API keys or network required (fakes + in-memory Qdrant)
```

Tooling is **ruff + mypy clean, CI wired**. The whole pipeline is built from small,
single-responsibility modules under `src/accurag/`; `pipeline.py` is the only file that
wires them together. Heavy SDKs are imported lazily, so `import accurag` is cheap and
key-free.
</content>
</invoke>
