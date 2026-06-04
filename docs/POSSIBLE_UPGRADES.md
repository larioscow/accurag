# Possible Upgrades (gated on a harder corpus)

## Why this file exists

The eval numbers in [`EVAL_RESULTS.md`](EVAL_RESULTS.md) are strong — but the corpus
is 47 **well-structured academic PDFs**: clean section hierarchy, consistent
formatting, real text layers. That is close to a *best case* for retrieval, and it
almost certainly flatters the results.

Two retrieval-improving techniques were evaluated and showed **no benefit** here —
and both times the measured reason was the same: **first-stage retrieval is already
saturated** on this corpus.

- Hybrid recall is **flat at 0.947 from depth 5 to 50** — nothing relevant sits
  below rank 5 to recover.
- **24 of 25** golden questions already land the right document in the top 5; the
  single miss is a *corpus-absence* failure (a chart-only appendix that doesn't
  contain the answer), which no retrieval technique can fix.

On a **messier corpus** — support tickets, scanned/OCR'd docs, inconsistent web
pages, fragmented notes, mixed formats — first-stage recall would drop, and these
techniques could genuinely earn their keep. They are **not rejected; they are
parked behind a test.** The point of a RAG library is to measure on *your* data;
these are the upgrades to revisit when the data gets hard.

## Prerequisite for all of the below: a harder-corpus benchmark

Before implementing any upgrade here, build or obtain a harder corpus + golden set
and measure the **gate**:

1. Run `strategy="hybrid"` recall@{5, 10, 20, 50} (the script pattern is in
   `scripts/` — the recall-by-depth probe).
2. **If recall is flat and high (>~0.95)** → retrieval isn't the bottleneck; these
   upgrades will do nothing. Spend effort on corpus coverage / answer quality instead.
3. **If recall is still rising at 20–50, or recall@5 is well below recall@50** →
   there is headroom. Implement the relevant upgrade and measure the delta.

Add a **chunk-level golden** for the hard corpus (see
`scripts/build_golden_chunk_labels.py`) so ranking/precision gains become visible —
doc-level labels hide them.

---

## Upgrade 1 — Deeper rerank candidate pool ("send more chunks to the reranker")

**What.** Over-fetch more hybrid candidates (e.g. 50–100) before reranking, instead
of 20. Already wired: `settings.rerank_candidates` / `retrieve(rerank_candidates=N)`.
Just change the number and measure — no new code.

**Why it didn't help here.** Recall is flat across depth, so a deeper pool only
hands the reranker more distractors. Measured: `hybrid_rerank` Recall@5 went
**0.933 → 0.893** going 20 → 50 (slightly *worse*).

**When it helps.** First-stage recall still *climbing* past 20 (relevant docs sitting
deep in the candidate list) **and** a reranker strong enough to sift the bigger pool.
This is the regime the *"From BM25 to Corrective RAG"* paper (doc 16) measured on
its harder text-and-table corpus.

**Test.** On the hard corpus, sweep `rerank_candidates ∈ {20, 50, 100}`; adopt the
best by Recall@5 / MRR. One-line change per run.

---

## Upgrade 2 — Contextual Retrieval

**What.** At index time, an LLM writes a short "how this chunk fits the document"
blurb and prepends it to each chunk before embedding (Anthropic's technique). Cost:
~**$1–2 one-time** with prompt caching, **zero per-query**. The corpus already
contains its results appendix (doc 41).

**Why it didn't help here.** 24/25 questions already retrieve the right doc in the
top 5 (no recall headroom); the one miss is corpus-absence, unfixable by changing
embeddings. No failure mode for it to address.

**When it helps.** Corpora full of chunks that are **ambiguous without document
context** — e.g. "the model scored 89.5" (which model? which task?), a ticket reply
that only makes sense given the thread, a table row detached from its caption. Common
in fragmented / conversational / poorly-structured data. Anthropic reports 35–49%
retrieval-failure reductions on such corpora.

**Test.** Two ingestion configs — with and without contextual blurbs — on the hard
corpus; compare recall **and** answer faithfulness on a chunk-level golden. Use a
cheap model (gpt-4o-mini / Haiku) for the blurbs.

---

## Upgrade 3 — HyDE (narrow fit; a long-tail fallback at best)

**What.** At query time, the LLM writes a hypothetical answer passage, embeds *that*,
and retrieves with its embedding — document-to-document similarity instead of
short-query-to-document (doc 11, *Precise Zero-Shot Dense Retrieval without Relevance
Labels*). Would drop in as `strategy="hyde"`.

**What the paper actually claims (read this before building it).** HyDE's value is the
**no-relevance-labels / cold-start** regime: it approximates a fine-tuned retriever
when you have *no labels yet*. As a search log accumulates labels and a supervised
dense retriever is trained, the supervised retriever takes over, and HyDE is relegated
to **rare and emerging queries**. HyDE and label-based retrieval serve *different query
populations*, not the same queries.

**How that maps to us — and why it's the weakest fit of the three.**
- We are permanently label-free (accurag never fine-tunes embeddings — no-GPU,
  hosted-APIs only), so HyDE's regime is technically always "on". *But* its gains were
  measured against weak 2022 zero-shot retrievers; we use `text-embedding-3-large`, a
  strong modern zero-shot embedder, and retrieval here is already saturated. The
  asymmetry HyDE fixes is largely handled by the base model.
- **Cost lands in the worst place.** Unlike contextual retrieval (index-time, free per
  query, deterministic), HyDE puts an **LLM call on the retrieval path, per query** —
  latency, per-query cost, and **non-determinism** (the hypothetical doc varies; even
  at temperature 0 it's model-version-sensitive). That cuts against the simple, fast,
  deterministic core this library is meant to be.

**When it's worth testing.** A *weak / un-fine-tuned* base embedder, or specifically
the **long-tail / rare / emerging** queries a strong embedder fumbles — not as a
blanket default. If tested, **segment the eval by query type**; per the paper, any
win shows up in the uncommon queries, not the average.

**Test.** Same harder-corpus gate, plus a query-type split. Treat HyDE as a *routed
fallback* for hard queries, not a replacement strategy — and weigh the per-query
latency/non-determinism against any recall gain before adopting.

---

## Upgrade 4 — Late chunking (double-blocked; the hardest to build)

**What.** Embed the *whole* document with a long-context model to get token-level
embeddings, then mean-pool them into per-chunk embeddings (doc 27). Each chunk's
embedding "knows" its surrounding context — same goal as contextual retrieval, but
architecturally (via pooling) instead of via LLM-generated text.

**The hard prerequisite (grounded, `[27-8]`/`[27-15]`).** It needs a long-context
embedding model that exposes **token-level** embeddings and uses **mean pooling**.
Our hosted **OpenAI** embedder returns a single pooled vector per input — no
token-level access, so it *cannot* do this. Our local **bge-small** is 512-context,
not long-context. So late chunking is **not a config change** — it requires a new
local long-context embedding backend (jina-v2/v3, nomic-embed, …) plus the pooling
logic, which cuts against the hosted-API / no-GPU / OpenAI-default design.

**Why no headroom here.** Same context-loss failure mode as contextual retrieval, and
retrieval is saturated — nothing to fix. The paper itself notes gains are
*"particularly for small chunk sizes"* `[27-30]`, and that large-chunk naive
sometimes wins; ours are 512 and structure-aware.

**Verdict.** Worst of the parked upgrades to justify — highest implementation cost
*and* no measurable headroom. Master plan already flagged it "highest complexity,
unproven; skip," now confirmed against the corpus. Revisit **only** if a local
long-context embedder is adopted for other reasons *and* the harder-corpus gate shows
real recall headroom.

## The honest meta-point

Two retrieval techniques in a row came back *"no headroom"* — not because they're
bad, but because **this corpus makes retrieval easy**. A RAG library's retrieval
metrics are only as informative as the corpus they're measured on. These upgrades
are real, standard, and well-supported by the literature; whether they help is a
property of the data, which is exactly why they belong behind a harder-corpus test
rather than shipped on faith. The most likely place real headroom lives on *this*
corpus is not retrieval at all — it's **corpus coverage** (figure/chart content the
parser can't read) and **answer quality**.
