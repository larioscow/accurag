"""Prompt construction for the grounded RAG answer step.

Design notes:
- Pure Python — no external SDK imports, no API keys required.
- ``build_grounded_prompt`` formats a numbered context window and instructs the
  model to answer ONLY from the context and say "I don't know" otherwise. It
  asks for **clean prose with no citation markers** — provenance is attached
  deterministically from the retrieved set (``Answer.sources``), so the answer
  text stays parse-free for the consumer.
- PROMPT_VERSION is bumped whenever the template changes so eval runs
  can be linked back to the exact prompt that produced them.
"""

from __future__ import annotations

from accurag.models import RetrievedChunk

PROMPT_VERSION = "v2"

_SYSTEM_INSTRUCTIONS = """\
You are a precise, grounded assistant. You must follow these rules:
1. Answer ONLY using information from the numbered context blocks provided below.
2. If the answer to the question is not present in the provided context, respond
   with exactly: "I don't know."
3. Do NOT use any prior knowledge or make assumptions beyond what the context
   explicitly states.
4. Write the answer as clean prose. Do NOT add citation markers, chunk ids, or
   bracketed references — the sources are tracked separately.
"""

_CONTEXT_BLOCK_TEMPLATE = """\
[{number}]
chunk_id: {chunk_id}
source_title: {source_title}
---
{text}
"""


def build_grounded_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    """Build a grounded prompt for the LLM.

    Each retrieved chunk is formatted as a numbered context block showing its
    ``chunk_id``, ``source_title``, and ``text``.  System instructions require
    the model to:
      - answer ONLY from the provided context,
      - respond "I don't know" when the answer is absent,
      - write clean prose with no citation markers (provenance is attached
        deterministically from the retrieved set).

    Args:
        query:     The user's question.
        retrieved: Ordered list of retrieved chunks (rank 0 = most relevant).

    Returns:
        A fully-formatted prompt string ready to pass to an LLM.
    """
    context_blocks: list[str] = []
    for i, rc in enumerate(retrieved, start=1):
        block = _CONTEXT_BLOCK_TEMPLATE.format(
            number=i,
            chunk_id=rc.chunk.chunk_id,
            source_title=rc.chunk.source_title,
            text=rc.chunk.text,
        )
        context_blocks.append(block)

    context_section = (
        "\n".join(context_blocks)
        if context_blocks
        else "(No context was retrieved for this query.)"
    )

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n"
        f"=== CONTEXT ===\n"
        f"{context_section}\n"
        f"=== QUESTION ===\n"
        f"{query}\n"
        f"=== ANSWER ===\n"
    )
