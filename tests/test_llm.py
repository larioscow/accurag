"""Tests for accurag.llm: hermetic, no API keys, no network.

Each provider (AnthropicLLM, OpenAILLM) now returns plain answer text. There is
no tool use or structured citation schema. Fakes return canned text.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

EXPECTED_ANSWER = "RAG improves factual grounding."


# ---------------------------------------------------------------------------
# Fake Anthropic client: messages.create -> content=[text block]
# ---------------------------------------------------------------------------


class _FakeAnthropicMessages:
    def create(self, **kwargs: Any):
        block = SimpleNamespace(type="text", text=EXPECTED_ANSWER)
        return SimpleNamespace(content=[block], stop_reason="end_turn")


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


# ---------------------------------------------------------------------------
# Fake OpenAI client: chat.completions.create -> choices[0].message.content
# ---------------------------------------------------------------------------


class _FakeOpenAIChatCompletions:
    def create(self, **kwargs: Any):
        msg = SimpleNamespace(content=EXPECTED_ANSWER)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAIChatCompletions())


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------


def test_anthropic_answer_returns_text():
    from accurag.llm import AnthropicLLM

    out = AnthropicLLM(client=FakeAnthropicClient()).answer("What is RAG?")
    assert isinstance(out, str)
    assert out == EXPECTED_ANSWER


def test_anthropic_default_model_is_claude():
    from accurag.llm import AnthropicLLM

    assert "claude" in AnthropicLLM().model.lower()


class _FakeAnthropicEmpty:
    """No text block (e.g. a refusal / max_tokens truncation)."""

    class _M:
        def create(self, **kwargs: Any):
            return SimpleNamespace(content=[], stop_reason="max_tokens")

    def __init__(self) -> None:
        self.messages = self._M()


def test_anthropic_empty_completion_raises_not_silent():
    import pytest

    from accurag.llm import AnthropicLLM, EmptyCompletionError

    assert issubclass(EmptyCompletionError, RuntimeError)  # backwards-compatible
    with pytest.raises(EmptyCompletionError, match="no answer text"):
        AnthropicLLM(client=_FakeAnthropicEmpty()).answer("q")


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------


def test_openai_answer_returns_text():
    from accurag.llm import OpenAILLM

    out = OpenAILLM(client=FakeOpenAIClient()).answer("What is RAG?")
    assert isinstance(out, str)
    assert out == EXPECTED_ANSWER


def test_openai_default_model_is_gpt():
    from accurag.llm import OpenAILLM

    assert "gpt" in OpenAILLM().model.lower()


class _FakeOpenAINone:
    """message.content is None (refusal / length truncation)."""

    class _C:
        def create(self, **kwargs: Any):
            msg = SimpleNamespace(content=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self._C())


def test_openai_empty_completion_raises_not_silent():
    import pytest

    from accurag.llm import EmptyCompletionError, OpenAILLM

    with pytest.raises(EmptyCompletionError, match="no answer text"):
        OpenAILLM(client=_FakeOpenAINone()).answer("q")


# ---------------------------------------------------------------------------
# Lazy-import guard
# ---------------------------------------------------------------------------


def test_module_importable_without_sdks():
    import importlib
    import sys

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("accurag.llm"):
            del sys.modules[mod_name]

    mod = importlib.import_module("accurag.llm")
    assert hasattr(mod, "AnthropicLLM")
    assert hasattr(mod, "OpenAILLM")
