"""Tests for accurag.cli — hermetic (no pipeline construction with real clients)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from accurag.cli import _cmd_ask, _load_golden, _make_pipeline, build_parser
from accurag.llm import EmptyCompletionError
from accurag.models import GoldenQA

_ROW = {"question": "q?", "ground_truth": "gt", "relevant_chunk_ids": ["1"]}


def test_load_golden_reads_jsonl(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(_ROW) for _ in range(3)) + "\n")
    golden = _load_golden(p)
    assert len(golden) == 3
    assert all(isinstance(g, GoldenQA) for g in golden)


def test_load_golden_reads_json_array(tmp_path):
    p = tmp_path / "golden.json"
    p.write_text(json.dumps([_ROW, _ROW]))
    assert len(_load_golden(p)) == 2


def test_load_golden_ignores_blank_lines(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text(json.dumps(_ROW) + "\n\n" + json.dumps(_ROW) + "\n")
    assert len(_load_golden(p)) == 2


def test_make_pipeline_defaults_inject_nothing():
    """Default flags must not construct heavy clients (embed_client/llm stay None)."""
    args = SimpleNamespace(embedder="openai", llm=None)
    pipe = _make_pipeline(args)
    assert pipe._embed_client is None
    assert pipe._llm is None


def test_cmd_ask_handles_empty_completion_cleanly(monkeypatch, capsys):
    """A refused/empty completion must exit non-zero with a message, not a raw
    traceback (the empty-completion guard raises; the CLI owns user-facing errors)."""
    from accurag import cli

    class _Pipe:
        def ask(self, *a, **k):
            raise EmptyCompletionError("no answer text (model=x, ...)")

    monkeypatch.setattr(cli, "_make_pipeline", lambda args: _Pipe())
    args = SimpleNamespace(query="q", strategy="dense", k=5, embedder="openai", llm=None)
    rc = _cmd_ask(args)
    assert rc == 1
    assert "No answer" in capsys.readouterr().err


def test_parser_exposes_embedder_and_llm_flags():
    parser = build_parser()
    a = parser.parse_args(["ingest", "--embedder", "local"])
    assert a.embedder == "local"
    a = parser.parse_args(["ask", "x", "--llm", "openai", "--embedder", "local"])
    assert a.llm == "openai" and a.embedder == "local"
    a = parser.parse_args(["evaluate", "g.jsonl", "--strategies", "dense,hybrid"])
    assert a.strategies == "dense,hybrid"
