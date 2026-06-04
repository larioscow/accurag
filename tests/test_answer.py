"""Tests for accurag.answer: hermetic, no API keys, fake LLM injected."""

from __future__ import annotations

from accurag.answer import build_answer
from accurag.models import Answer, Chunk, RetrievedChunk


def _rc(chunk_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=1,
            text=text,
            source_title="Lost in the Middle",
            theme="retrieval",
            section="Intro",
            url="https://example.com/paper",
        ),
        score=1.0 - rank * 0.1,
        rank=rank,
    )


class _FakeLLM:
    """An LLM whose .answer() returns fixed text and records the prompt."""

    def __init__(self, text: str) -> None:
        self._text = text

    def answer(self, prompt: str) -> str:
        self.seen_prompt = prompt
        return self._text


def test_sources_are_the_retrieved_chunks_deterministically() -> None:
    retrieved = [_rc("1-0", "a", 0), _rc("1-1", "b", 1)]
    llm = _FakeLLM("Models lose track of the middle.")

    answer = build_answer("what happens in the middle?", retrieved, llm)

    assert isinstance(answer, Answer)
    assert answer.text == "Models lose track of the middle."
    # sources are exactly the retrieved set: typed and complete, taken from retrieval rather than the LLM's report
    assert [s.chunk.chunk_id for s in answer.sources] == ["1-0", "1-1"]
    assert answer.sources[0].chunk.source_title == "Lost in the Middle"
    assert answer.sources[0].chunk.url == "https://example.com/paper"
    assert answer.sources[0].score == 1.0
    # the prompt actually reached the LLM
    assert "1-0" in llm.seen_prompt


def test_answer_text_is_stripped() -> None:
    answer = build_answer("q", [_rc("1-0", "x", 0)], _FakeLLM("  spaced out  \n"))
    assert answer.text == "spaced out"


def test_no_retrieval_means_no_sources() -> None:
    answer = build_answer("q", [], _FakeLLM("I don't know."))
    assert answer.text == "I don't know."
    assert answer.sources == []
