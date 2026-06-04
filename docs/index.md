# accurag

A lean, evaluation-driven RAG **library** — not a deployed app, not a framework. You
`import accurag` (or drive the CLI) and get four verbs — **ingest / retrieve / ask /
evaluate** — over one strategy switch: `dense | hybrid | hybrid_rerank`. That switch
*is* the comparison the eval measures.

## Why it exists: the evaluation

The reason this library exists is the evaluation. Which parser, which retrieval
strategy, what chunk size — every one of those was decided by running the eval, not by
argument. And the sources an answer returns are **the retrieved chunks themselves,
assigned in Python** — provenance is deterministic, never self-reported by the model.

That kills the *hallucinated / nonexistent-citation* class entirely. It does **not**
catch a wrong answer drawn from a real-but-bad chunk: the source is guaranteed real and
guaranteed to be the one used, not guaranteed to have been read correctly.

## Headline result

Three retrieval strategies, scored on a hand-built golden set of **25 Q&A** over all 47
documents (Docling parser, 4,215 chunks). Faithfulness and answer relevancy are graded
by an in-library LLM judge (`gpt-4o-mini`), which is **not** the model writing the
answers (Claude).

| Config | Faithfulness | Answer Rel. | Recall@5 | MRR | P95 Latency (ms)* |
|---|---|---|---|---|---|
| dense (naive) | 0.988 | 0.776 | 0.867 | 0.803 | 670 |
| hybrid (dense+sparse, RRF) | 0.978 | 0.776 | **0.947** | 0.828 | 454 |
| hybrid + rerank (Cohere) | 0.984 | **0.808** | 0.933 | **0.861** | 1208 |

\* Rerank's 1208 ms is inflated by a Cohere **trial key** (10 calls/min, backoff under
load), not the reranker itself; treat dense/hybrid (~0.5 s) as representative and rerank
as an upper bound. Latency is indicative, not benchmarked.

The most useful result is a *negative* one: deepening the reranker's candidate pool
20 → 50 was **measured** and **slightly hurt** recall@5 (0.933 → 0.893), so it wasn't
shipped. The mechanism is quantified — hybrid first-stage recall is flat at 0.947 from
depth 5 to 50, so a deeper pool only feeds the reranker distractors. Contextual
retrieval and late chunking were *not* measured; they were predicted to have no headroom
and parked. "Measured" is claimed only for rerank-depth.

!!! note "Honest framing: the corpus is easy, and recall is saturated"
    The shipped corpus is **~47 clean AI/RAG technical papers and vendor docs** — clean
    academic PDFs, which makes retrieval easy. First-stage recall is **saturated**
    (0.947, flat with depth), which is *why* the retrieval upgrades above show little
    headroom *here*. They may well help on messier data. Note also that the 0.947 is
    **doc-level** recall; with stricter chunk-level labels recall@5 drops to ~0.53,
    because there are ~4.9 relevant chunks/question against only k=5 slots (recall@5 is
    mathematically capped when |relevant| > k). Both label sets exist and each is used
    where it's valid.

## Where to next

- **[Getting started](getting-started.md)** — install, the four verbs, the CLI, and
  no-API mode (`--embedder local` runs ingest + retrieve with no key or network).
- **[Architecture](architecture.md)** — the two halves joined only by the Qdrant index,
  and the deterministic-provenance design.
- **[Evaluation results](EVAL_RESULTS.md)** — the full breakdown: doc- vs chunk-level
  labels, the chunk-size sweep, the parser A/B, and the rerank-depth non-result.
- **[API reference](reference/accurag/index.md)** — the typed public surface.
