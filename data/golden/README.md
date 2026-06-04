# Golden Evaluation Set

**File:** `golden.jsonl` — 24 hand-written GoldenQA records used to score retrieval and generation quality across retrieval strategies.

---

## Format

Each line is a JSON object with three fields, matching the `GoldenQA` Pydantic model in `src/accurag/models.py`:

```json
{
  "question": "string — the natural-language question",
  "ground_truth": "string — the expected correct answer",
  "relevant_chunk_ids": ["list", "of", "string", "IDs"]
}
```

### IMPORTANT: `relevant_chunk_ids` are doc-level placeholders

At authoring time, exact chunk IDs (of the form `{doc_id}-{ordinal}`) do not yet exist because the corpus has not been ingested. The strings stored in `relevant_chunk_ids` are the **manifest document `id` values** (integers serialized as strings) of the document(s) that contain the answer.

**After the first ingestion run**, these must be refined:

1. Run ingestion with `index_chunks` to populate Qdrant.
2. For each golden question, retrieve with `dense()` or `hybrid()` and identify the specific chunk IDs (e.g. `"3-4"`, `"12-1"`) that actually contain the answer span.
3. Replace the doc-id placeholders with exact chunk IDs.
4. Re-commit `golden.jsonl` with a note: `"golden: refine chunk_ids post-ingestion"`.

Until that refinement, recall@k and MRR metrics computed against this file measure **document-level** hit rate, not chunk-level. That is still a useful signal during early development.

---

## The Four Question Categories

The 24 questions are spread across four categories, each testing a distinct hypothesis about retrieval quality:

### (a) Exact-Term Lookups — 8 questions

**Questions:** RRF formula, DPR dual-encoder, ColBERT late interaction, HyDE mechanism, Docling TableFormer, SPLADE sparse vectors, Ragas metric definitions, BM42 vs BM25.

**Hypothesis:** A retrieval system must be able to surface the correct single document when the question uses the exact named term (RRF, HyDE, SPLADE, ColBERT) that appears in only one paper. Dense retrieval should handle these via semantic matching; sparse/BM25-style retrieval may have an edge on exact acronyms.

**Tested failure mode:** Dense-only retrieval returning a semantically close but wrong paper (e.g. returning DPR when asked about ColBERT).

---

### (b) Numeric / Table Values — 4 questions

**Questions:** ColBERTv2 residual compression storage reduction, ReWOO 5x token reduction, Contextual Retrieval failure-rate numbers, 2026 chunking study finding on overlap cost.

**Hypothesis:** Numeric claims (benchmark tables, specific percentages, "5x") are hardest for retrieval because the number rarely appears verbatim in an abstract — it lives in a results section or table. Structure-aware chunking that preserves table context should outperform naive fixed-size chunking on these.

**Tested failure mode:** The retrieved context includes the right paper but the wrong section, returning the method description rather than the results table with the number.

---

### (c) Near-Neighbor Disambiguation — 6 questions

**Questions:** REALM vs RAG foundational distinction, FLARE vs standard RAG retrieval timing, ARES vs Ragas evaluation approach, Reflexion vs RL gradient updates, Self-Route vs naive RAG vs long-context, GraphRAG and why it is excluded.

**Hypothesis:** The corpus contains many papers on closely related topics (multiple agent papers, multiple eval frameworks, multiple hybrid-retrieval papers). A good retrieval system must return the *right* paper when the question requires distinguishing between two similar concepts. Hybrid retrieval with reranking should outperform pure-dense on these because lexical signals (paper names, author names) help.

**Tested failure mode:** Returning Reflexion when asked about ReAct, or returning Ragas when asked about ARES, because they share dense-space neighborhood.

---

### (d) Broad Conceptual Controls — 6 questions

**Questions:** Dense vs sparse retrieval conceptual difference, Naive/Advanced/Modular RAG taxonomy, "Lost in the Middle" U-shaped position bias, "Correctness is not Faithfulness" failure mode, Self-RAG self-reflection mechanism, DocLayNet document categories.

**Hypothesis:** These questions have wide-coverage answers that may be spread across multiple chunks or documents. A good system should retrieve context from multiple relevant documents (multi-doc recall). These questions also serve as sanity checks: if context precision is low on these, the retrieval is noisy.

**Tested failure mode:** Returning only one relevant document when the answer synthesizes across two or three. Also tests that the system does NOT hallucinate beyond what is in the retrieved context.

---

## Coverage Across Corpus Themes

| Theme | Doc IDs Covered | # Questions |
|---|---|---|
| `rag_foundations` | 2, 3, 4, 6 | 5 |
| `retrieval_rerank` | 8, 9, 10, 11, 12, 13 | 7 |
| `agentic_patterns` | 17, 19, 23, 24 | 4 |
| `chunking_parsing` | 28, 31, 32 | 3 |
| `evaluation_faithfulness` | 33, 34, 40 | 3 |
| `vendor_reports` | 41, 43, 46 | 3 |
| Cross-theme (dense+sparse+RRF) | 8, 12, 13 | 1 (multi-doc) |

---

## Eval Harness Integration

The `EvalReport.to_markdown()` method (see `src/accurag/models.py`) renders the comparison table. The golden set feeds into Ragas metrics:

- **Context Precision / Context Recall**: compare retrieved chunk IDs against `relevant_chunk_ids`.
- **Faithfulness**: verify LLM answer claims are supported by retrieved context.
- **Answer Relevancy**: LLM-judged match between generated answer and `question`.
- **Recall@k / MRR**: rank-based metrics against `relevant_chunk_ids`.

Three strategies are evaluated head-to-head:

| Config | Description |
|---|---|
| `dense` | OpenAI `text-embedding-3-large` + cosine ANN, no reranking |
| `hybrid` | Dense + BM42 sparse, RRF fusion via Qdrant Query API |
| `hybrid+rerank` | Hybrid retrieval + Cohere reranker on top-20 |

The comparison table is the primary deliverable of the eval harness.
