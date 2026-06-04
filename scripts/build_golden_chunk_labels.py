"""Build CHUNK-LEVEL relevance labels for the golden set (TREC-style pooling).

For each golden question:
  1. Pool candidate chunks = union of the top-k retrieved by `dense` and `hybrid`
     (pooling multiple retrievers reduces single-retriever bias).
  2. LLM-judge (gpt-4o-mini) whether each pooled chunk actually contains
     information that helps answer the question (grounded in the reference answer).
  3. Write the relevant chunk_ids.

Output: data/golden/golden_chunks.jsonl — same GoldenQA schema, but
relevant_chunk_ids hold CHUNK ids (e.g. "12-3"). These are **index-specific**
(they reference the chunking of whatever index this is run against), so the
portable doc-level data/golden/golden.jsonl is kept for cross-parser comparison.

Usage:  python scripts/build_golden_chunk_labels.py [collection]   (default: accurag_docling)
Cost:   ~25 questions x ~18 pooled chunks = a few hundred gpt-4o-mini calls (~$0.05-0.10).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import OpenAI

from accurag import GoldenQA, RagPipeline
from accurag.config import settings

COLLECTION = sys.argv[1] if len(sys.argv) > 1 else "accurag_docling"
POOL_K = 12
JUDGE_MODEL = "gpt-4o-mini"

_JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relevant": {
            "type": "boolean",
            "description": "True iff the CHUNK contains information that helps answer the QUESTION.",
        }
    },
    "required": ["relevant"],
}


def is_relevant(judge: OpenAI, question: str, reference: str, chunk_text: str) -> bool:
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CHUNK:\n{chunk_text}\n\n"
        "Does the CHUNK contain information that helps answer the QUESTION? "
        "Judge by content overlap with the question/reference, not writing style."
    )
    resp = judge.chat.completions.create(
        model=JUDGE_MODEL,
        max_tokens=20,
        tools=[{"type": "function", "function": {
            "name": "judge", "parameters": _JUDGE_SCHEMA, "strict": True}}],
        tool_choice={"type": "function", "function": {"name": "judge"}},
        messages=[{"role": "user", "content": prompt}],
    )
    args = resp.choices[0].message.tool_calls[0].function.arguments
    return bool(json.loads(args)["relevant"])


def main() -> None:
    settings.collection = COLLECTION
    golden = [
        GoldenQA(**json.loads(line))
        for line in Path("data/golden/golden.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rag = RagPipeline()
    judge = OpenAI(api_key=settings.openai_api_key or None)

    out: list[dict] = []
    for qa in golden:
        pool: dict[str, str] = {}
        for strategy in ("dense", "hybrid"):
            for rc in rag.retrieve(qa.question, strategy=strategy, k=POOL_K):
                pool[rc.chunk.chunk_id] = rc.chunk.text
        relevant = [
            cid for cid, text in pool.items()
            if is_relevant(judge, qa.question, qa.ground_truth, text)
        ]
        rec = qa.model_dump()
        # Fall back to the doc-level label if the judge finds nothing (don't drop the question).
        rec["relevant_chunk_ids"] = relevant or qa.relevant_chunk_ids
        out.append(rec)
        print(f"[label] {len(relevant):>2}/{len(pool):>2} relevant  ::  {qa.question[:60]}", file=sys.stderr, flush=True)

    dest = Path("data/golden/golden_chunks.jsonl")
    dest.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    n_chunk_level = sum(1 for r in out if any("-" in cid for cid in r["relevant_chunk_ids"]))
    print(f"[label] wrote {dest} ({n_chunk_level}/{len(out)} questions got chunk-level labels)")


if __name__ == "__main__":
    main()
