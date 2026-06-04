# CLI reference

`accurag` ships a thin command-line wrapper over the same four verbs the library exposes.
It has three subcommands — `ingest`, `ask`, and `evaluate` — each mapping onto the
pipeline. Heavy SDKs are imported lazily, so `python -m accurag.cli --help` works with no
API keys installed.

Invoke it as a module:

```bash
python -m accurag.cli <subcommand> [flags]
```

!!! note "Keys depend on what you run"
    The CLI honours the same key rules as the library. `--embedder local` (fastembed,
    CPU-only) lets **ingest + retrieve** run with no key and no network. Answering
    (`ask`) still needs a hosted LLM — generation is the one part `--embedder local`
    can't cover.

    | Variable | Used for | Required? |
    |---|---|---|
    | `OPENAI_API_KEY` | embeddings + the eval judge | no — `--embedder local` avoids it |
    | `ANTHROPIC_API_KEY` | answer generation (Claude) | for `ask`; or pass `--llm openai` |
    | `COHERE_API_KEY` | reranking | only for `hybrid_rerank` (a free trial key works) |

The `--embedder` flag is shared by all three subcommands. It must be **the same for
ingest and query** — a `local`-embedded index can't be queried with `openai` embeddings
and vice versa.

| Flag | Choices | Default | Notes |
|---|---|---|---|
| `--embedder` | `openai`, `local` | `openai` | `local` = fastembed, no API key/network. |

---

## `ingest`

Build the Qdrant index from the corpus manifest: parse → chunk → embed → index.

| Flag | Type / choices | Default | Notes |
|---|---|---|---|
| `--manifest` | path | `settings.manifest_path` | Path to the corpus manifest JSON. |
| `--limit` | int | `None` | Only ingest the first N manifest entries. |
| `--parser` | `docling`, `pymupdf` | `docling` | `pymupdf` = fast/low-memory text extraction (no ML, seconds); `docling` = ML layout+table parsing (slow, GBs). |
| `--chunk-size` | int | `None` → `settings.chunk_size` (512) | Target chunk size in tokens, clamped to the embedder's window. |
| `--vision` | flag | off | Also read figures/charts into text chunks via a vision model (default Claude Sonnet; set `ACCURAG_VISION_MODEL` to override). PDFs only; extra parse + per-figure API cost. |
| `--embedder` | `openai`, `local` | `openai` | See shared flag above. |

```bash
python -m accurag.cli ingest --parser docling          # ~15 min CPU, ~$0.10 embeddings
```

!!! warning "`--vision` is numerically unreliable"
    With the cheap vision model, chart *structure* (titles, axes, series) reads reliably
    but *bar/line heights are guessed*, so a quantitative chart answer can come back
    grounded but numerically wrong. This is a documented limitation, not a feature.

---

## `ask`

Retrieve, then generate a grounded answer plus its sources for a single query. Prints the answer
text, then a `Sources:` block listing the retrieved chunks (chunk id, source title,
section, URL).

| Argument / flag | Type / choices | Default | Notes |
|---|---|---|---|
| `query` (positional) | string | — | The question to ask. |
| `--strategy` | `dense`, `hybrid`, `hybrid_rerank` | `dense` | Retrieval strategy. |
| `-k` | int | `5` | Number of chunks to retrieve. |
| `--embedder` | `openai`, `local` | `openai` | See shared flag above. |
| `--llm` | `anthropic`, `openai` | `None` → Anthropic Claude | Answer LLM. Use `openai` to run with an OpenAI chat key instead. |

```bash
python -m accurag.cli ask "How does reciprocal rank fusion work?" --strategy hybrid
```

`hybrid_rerank` needs a `COHERE_API_KEY` (a free trial key works). The sources printed
are the retrieved chunks themselves, assigned in Python — not citations reported by the
model.

---

## `evaluate`

Score one or more strategies over a golden set and print the results table as Markdown.
The golden file may be JSONL (one object per line) or a JSON array.

| Argument / flag | Type / choices | Default | Notes |
|---|---|---|---|
| `golden` (positional) | path | — | Path to the golden-set JSON file. |
| `-k` | int | `5` | Number of chunks to retrieve. |
| `--strategies` | comma-separated string | `dense,hybrid,hybrid_rerank` | Strategies to score. Use `dense,hybrid` to stay fully offline (`hybrid_rerank` needs a Cohere key). |
| `--answer-quality` | flag | off | Also grade faithfulness + answer relevancy with an LLM judge (generates answers + judge calls — costs API; judge = `gpt-4o-mini`). |
| `--embedder` | `openai`, `local` | `openai` | See shared flag above. |

```bash
python -m accurag.cli evaluate data/golden/golden.jsonl --answer-quality
```

Fully offline (no keys, no network) — ingest with `--embedder local` first, then:

```bash
python -m accurag.cli evaluate data/golden/golden.jsonl --embedder local --strategies dense,hybrid
```

The judge (`gpt-4o-mini`) is a different model from the generator (Claude), so grader and
graded aren't the same system. Its robustness — per-sample try/except plus a reported
`coverage` field — applies to the default in-library path; the optional Ragas path and
malformed inputs are not guarded.
