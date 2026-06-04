"""Run accurag with NO API keys, using local embeddings.

This exercises ingest -> dense + hybrid retrieval entirely offline via
`LocalEmbeddingClient` (fastembed dense) + the local SPLADE sparse model +
local-mode Qdrant. The optional grounded-answer step at the end only runs if an
OpenAI chat key is available (it is the one part that needs an LLM API).

Usage:
    python examples/local_offline_smoke.py            # ingest 2 docs + retrieve
    python examples/local_offline_smoke.py --limit 5  # ingest more docs

Nothing here costs money for the retrieval path; the corpus is small and the
models are cached after first download.
"""
from __future__ import annotations

import argparse

from accurag import LocalEmbeddingClient, RagPipeline
from accurag.config import settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2, help="docs to ingest")
    ap.add_argument("--query", default="What is retrieval-augmented generation?")
    args = ap.parse_args()

    # No API key needed: local dense embedder + (default) local sparse model.
    rag = RagPipeline(embed_client=LocalEmbeddingClient())

    n = rag.ingest(limit=args.limit)
    print(f"ingested {n} chunks from {args.limit} docs (fully offline)\n")

    for strategy in ("dense", "hybrid"):
        print(f"=== retrieve(strategy={strategy!r}) ===")
        for h in rag.retrieve(args.query, strategy=strategy, k=3):
            print(f"  [{h.rank}] {h.score:.3f} {h.chunk.chunk_id}  {h.chunk.source_title[:55]}")
        print()

    # The answer step is the ONLY part that needs an LLM API key.
    if settings.openai_api_key:
        from accurag.llm import OpenAILLM

        rag._llm = OpenAILLM(model="gpt-4o-mini")
        ans = rag.ask(args.query, strategy="hybrid", k=3)
        print("=== ask (OpenAI gpt-4o-mini) ===")
        print(ans.text)
        print("sources:", [s.chunk.chunk_id for s in ans.sources])
    else:
        print("(skipping `ask` — no OPENAI_API_KEY set; retrieval above is fully offline)")


if __name__ == "__main__":
    main()
