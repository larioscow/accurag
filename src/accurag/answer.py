"""Build a grounded Answer from retrieved context.

The model writes the answer text. The sources are the retrieved chunks,
attached in Python rather than parsed out of the model output, so a consumer
gets typed provenance in ``answer.sources``.
"""

from __future__ import annotations

from typing import Any

from accurag.models import Answer, RetrievedChunk
from accurag.prompts import build_grounded_prompt


def build_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    llm: Any,
) -> Answer:
    """Produce a grounded :class:`Answer` for *query* over *retrieved* context.

    Args:
        query:     The user's question.
        retrieved: Ordered retrieved chunks (rank 0 = most relevant).
        llm:       An object exposing ``.answer(prompt) -> str``.

    Returns:
        An :class:`Answer` with the generated ``text`` and ``sources`` set to the
        retrieved chunks the answer was conditioned on.
    """
    prompt = build_grounded_prompt(query, retrieved)
    text = llm.answer(prompt)
    return Answer(text=text.strip(), sources=list(retrieved))
