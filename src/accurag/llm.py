"""LLM providers for grounded answer generation.

Two providers, both returning plain answer text. Neither uses tool-use or a
structured citation schema. Sources are tracked in Python (the retrieved
chunks), so the model's only job is to write a grounded answer from the context.

- :class:`AnthropicLLM`: primary (Claude).
- :class:`OpenAILLM`: fallback (GPT-*).

Both SDKs are imported lazily inside methods so this module imports, and the
classes instantiate, without API keys or network. The SDK client is injectable
for hermetic testing.

Usage::

    from accurag.llm import AnthropicLLM
    llm = AnthropicLLM()
    text = llm.answer(prompt)   # -> str
"""

from __future__ import annotations

from typing import Any


class EmptyCompletionError(RuntimeError):
    """Raised when a provider returns no answer text (refusal / length truncation).

    Subclasses ``RuntimeError`` for backwards compatibility. This is the contract
    every ``.answer()`` upholds: it returns non-empty text or raises, and never
    returns ``""``. Callers that batch many calls (e.g. ``RagPipeline.evaluate``)
    catch this specifically so one empty completion doesn't abort the run, while
    a real network/API error still propagates.
    """


def _require_nonempty(text: str, *, provider: str, model: str, reason: str) -> str:
    """Return *text* if non-empty, else raise :class:`EmptyCompletionError`."""
    if text:
        return text
    raise EmptyCompletionError(
        f"{provider} returned no answer text (model={model}, {reason}), likely a "
        "refusal or length truncation, not a real empty answer."
    )


class AnthropicLLM:
    """Generate a grounded answer with Claude (plain text).

    Args:
        model: Claude model identifier.
        max_tokens: Maximum tokens for the completion.
        client: Injectable ``anthropic.Anthropic`` instance; when ``None`` a real
                client is built lazily from ``ANTHROPIC_API_KEY``.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 1024,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import anthropic

        from accurag.config import settings

        key = self._api_key if self._api_key is not None else settings.anthropic_api_key
        return anthropic.Anthropic(api_key=key or None)

    def answer(self, prompt: str) -> str:
        """Send *prompt* to Claude and return the answer text."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        ).strip()
        return _require_nonempty(
            text,
            provider="Anthropic",
            model=self.model,
            reason=f"stop_reason={getattr(response, 'stop_reason', '?')}",
        )


class OpenAILLM:
    """Generate a grounded answer with GPT-* (plain text).

    Args:
        model: OpenAI model identifier.
        max_tokens: Maximum tokens for the completion.
        client: Injectable ``openai.OpenAI`` instance; when ``None`` a real
                client is built lazily from ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai

        from accurag.config import settings

        key = self._api_key if self._api_key is not None else settings.openai_api_key
        return openai.OpenAI(api_key=key or None)

    def answer(self, prompt: str) -> str:
        """Send *prompt* to GPT-* and return the answer text."""
        response = self._get_client().chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        return _require_nonempty(
            text,
            provider="OpenAI",
            model=self.model,
            reason=f"finish_reason={getattr(response.choices[0], 'finish_reason', '?')}",
        )
