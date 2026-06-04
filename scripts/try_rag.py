#!/usr/bin/env python
"""Try the accurag RAG system against the indexed corpus.

    python scripts/try_rag.py                  # interactive — ask questions in a loop
    python scripts/try_rag.py "your question"  # one-shot

Queries the `accurag_docling` index (the 4,215-chunk Docling build) with hybrid
retrieval, generates a grounded answer, and prints the sources each citation
points to. Needs the index built (`python -m accurag.cli ingest --parser docling`)
and a key for answer generation (ANTHROPIC_API_KEY, or OPENAI_API_KEY as fallback).
"""
import logging
import os
import sys
import warnings

# Quiet the model-loading noise (HF hub chatter, tokenizer warnings) before the
# heavy imports — purely cosmetic, the work is identical with or without it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
for _noisy in ("huggingface_hub", "transformers", "fastembed", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from accurag import RagPipeline          # noqa: E402
from accurag.config import settings      # noqa: E402

COLLECTION = "accurag_docling"
STRATEGY = "hybrid"
K = 5

EXAMPLES = [
    "How does reciprocal rank fusion combine multiple ranked lists?",
    "What is retrieval-augmented generation?",
    "Do language models struggle to use information in the middle of long contexts?",
    "What is dense passage retrieval and how does it differ from BM25?",
]


def _pick_llm():
    """Use Claude if its key is set, else fall back to OpenAI."""
    if settings.anthropic_api_key:
        return None  # RagPipeline default is AnthropicLLM
    if settings.openai_api_key:
        from accurag.llm import OpenAILLM
        return OpenAILLM()
    sys.exit("No answer-generation key. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env.")


def _answer(rag: RagPipeline, question: str) -> None:
    print(f"\nQ: {question}\n   …retrieving + answering…\n")
    ans = rag.ask(question, strategy=STRATEGY, k=K)
    print(ans.text.strip())
    if ans.sources:
        print("\nSources (the retrieved chunks the answer was grounded on):")
        for s in ans.sources:
            c = s.chunk
            section = f" — {c.section}" if c.section else ""
            print(f"  [{c.chunk_id}] {c.source_title}{section}  (score {s.score:.2f})")
            print(f"        {c.url}")


def main() -> None:
    rag = RagPipeline(collection=COLLECTION, llm=_pick_llm())  # instance config — no global

    try:
        count = rag.qdrant.count(COLLECTION).count
    except Exception:
        count = 0
    if not count:
        sys.exit(
            f"No index in collection '{COLLECTION}'. Build one first:\n"
            f"    python -m accurag.cli ingest --parser docling"
        )

    print(f"accurag — index '{COLLECTION}' ({count} chunks), strategy={STRATEGY}")

    if len(sys.argv) > 1:
        _answer(rag, " ".join(sys.argv[1:]))
        return

    print("\nAsk a question (blank line to quit). Some it can answer well:")
    for e in EXAMPLES:
        print(f"  - {e}")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        _answer(rag, q)


if __name__ == "__main__":
    main()
