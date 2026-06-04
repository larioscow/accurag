"""Tests for accurag.vision. Hermetic (fake vision clients, tiny PIL image, does not run Docling)."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from accurag.config import settings
from accurag.models import ManifestEntry
from accurag.vision import describe_figure, figure_chunks


def _tiny_image():
    return Image.new("RGB", (8, 8), "white")


class _FakeOpenAI:
    """Mimics openai chat.completions.create; records the request."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content="Bar chart: failure rate at 20.")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class _FakeClaude:
    """Mimics anthropic messages.create; records the request."""

    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Grid chart: 5 panels, Voyage/Cohere/Gemini.")
            ]
        )


def _entry():
    return ManifestEntry(
        id=41,
        title="Contextual Retrieval Appendix",
        authors="a",
        year=2024,
        arxiv_id=None,
        pdf_url="http://x",
        source="vendor",
        theme="rag",
        has_tables_or_figures=True,
        verified=True,
    )


def test_default_vision_model_is_claude():
    assert settings.vision_model.startswith("claude")


def test_describe_figure_default_uses_anthropic_image_block():
    client = _FakeClaude()
    out = describe_figure(_tiny_image(), client=client)  # no model -> default (claude)
    assert "Grid chart" in out
    content = client.calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["data"]  # base64 payload present


def test_describe_figure_openai_path_when_model_is_gpt():
    client = _FakeOpenAI()
    out = describe_figure(_tiny_image(), client=client, model="gpt-4o-mini")
    assert "failure rate" in out
    content = client.calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class _FakeOpenAINoneContent:
    """OpenAI returning message.content=None (refusal / max_tokens truncation)."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])


class _FakeClaudeLeadingNonText:
    """Claude returning a leading non-text block (e.g. a thinking block) then text."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        thinking = SimpleNamespace(type="thinking")  # no .text attribute
        text = SimpleNamespace(type="text", text="Real description.")
        return SimpleNamespace(content=[thinking, text])


def test_describe_figure_openai_none_content_returns_empty_not_crash():
    # content=None must not raise AttributeError; caller treats "" as "skip figure".
    assert (
        describe_figure(_tiny_image(), client=_FakeOpenAINoneContent(), model="gpt-4o-mini") == ""
    )


def test_describe_figure_claude_skips_leading_non_text_block():
    out = describe_figure(_tiny_image(), client=_FakeClaudeLeadingNonText())
    assert out == "Real description."  # thinking block ignored, no crash


def test_figure_chunks_skips_empty_descriptions(monkeypatch):
    import accurag.vision as vmod

    monkeypatch.setattr(vmod, "extract_figure_images", lambda path: [_tiny_image()])
    # a None-content vision response yields "" -> that figure is skipped, no crash
    chunks = figure_chunks(
        "ignored.pdf", _entry(), client=_FakeOpenAINoneContent(), model="gpt-4o-mini"
    )
    assert chunks == []


def test_figure_chunks_builds_chunks(monkeypatch):
    import accurag.vision as vmod

    monkeypatch.setattr(vmod, "extract_figure_images", lambda path: [_tiny_image(), _tiny_image()])

    chunks = figure_chunks("ignored.pdf", _entry(), client=_FakeClaude())
    assert [c.chunk_id for c in chunks] == ["41-fig0", "41-fig1"]
    assert all(c.doc_id == 41 for c in chunks)
    assert chunks[0].section == "Figure 1"
    assert chunks[0].text.startswith("[Figure 1]")
