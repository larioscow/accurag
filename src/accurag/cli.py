"""Command-line entry point for accurag.

Run via ``python -m accurag.cli <subcommand> ...``.

Subcommands
-----------
ingest    Build the Qdrant index from the corpus manifest.
ask       Retrieve + generate a grounded, cited answer for a query.
evaluate  Score dense/hybrid/hybrid_rerank over a golden set and print the table.

Heavy SDKs are only imported when a subcommand actually needs them (the
pipeline builds its collaborators lazily), so ``python -m accurag.cli --help``
works with no API keys installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from accurag.config import settings
from accurag.models import GoldenQA


def _make_pipeline(args: argparse.Namespace):
    """Build a RagPipeline honouring --embedder / --llm flags.

    --embedder local  -> LocalEmbeddingClient (fastembed, no API key/network).
    --llm openai|anthropic -> pick the answer LLM; default (None) = pipeline
    default (Anthropic Claude). Embedder must match between ingest and query.
    """
    from accurag.pipeline import RagPipeline

    embed_client = None
    if getattr(args, "embedder", "openai") == "local":
        from accurag.embed import LocalEmbeddingClient

        embed_client = LocalEmbeddingClient()

    llm: Any = None
    chosen = getattr(args, "llm", None)
    if chosen == "openai":
        from accurag.llm import OpenAILLM

        llm = OpenAILLM()
    elif chosen == "anthropic":
        from accurag.llm import AnthropicLLM

        llm = AnthropicLLM()

    return RagPipeline(embed_client=embed_client, llm=llm)


def _cmd_ingest(args: argparse.Namespace) -> int:
    pipe = _make_pipeline(args)
    n = pipe.ingest(
        manifest_path=args.manifest,
        limit=args.limit,
        parser=args.parser,
        chunk_size=args.chunk_size,
        vision=args.vision,
    )
    print(f"indexed {n} chunks")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from accurag.llm import EmptyCompletionError

    pipe = _make_pipeline(args)
    try:
        answer = pipe.ask(args.query, strategy=args.strategy, k=args.k)
    except EmptyCompletionError as exc:
        print(f"No answer: {exc}", file=sys.stderr)
        return 1
    print(answer.text)
    print()
    print("Sources:")
    for s in answer.sources:
        c = s.chunk
        section = f", {c.section}" if c.section else ""
        print(f"  [{c.chunk_id}] {c.source_title}{section} ({c.url})")
    return 0


def _load_golden(path: Path) -> list[GoldenQA]:
    """Load a golden set from JSONL (one object per line) or a JSON array."""
    text = path.read_text(encoding="utf-8").strip()
    stripped = text.lstrip()
    if stripped.startswith("["):  # JSON array
        rows = json.loads(text)
    else:  # JSONL: one GoldenQA per non-blank line
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [GoldenQA(**item) for item in rows]


def _cmd_evaluate(args: argparse.Namespace) -> int:
    golden = _load_golden(Path(args.golden))

    pipe = _make_pipeline(args)
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    report = pipe.evaluate(
        golden, strategies=strategies, k=args.k, answer_quality=args.answer_quality
    )
    print(report.to_markdown())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accurag",
        description="Production RAG over a curated corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_embedder_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--embedder",
            choices=["openai", "local"],
            default="openai",
            help="Embedding backend. 'local' = fastembed, no API key/network "
            "(must be the same for ingest and query).",
        )

    p_ingest = sub.add_parser("ingest", help="Build the Qdrant index from the manifest.")
    p_ingest.add_argument(
        "--manifest",
        type=Path,
        default=settings.manifest_path,
        help="Path to the corpus manifest JSON.",
    )
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N manifest entries.",
    )
    p_ingest.add_argument(
        "--parser",
        choices=["docling", "pymupdf"],
        default="docling",
        help="Document parser. 'pymupdf' = fast/low-memory text extraction "
        "(no ML, seconds); 'docling' = ML layout+table parsing (slow, GBs).",
    )
    p_ingest.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Target chunk size in tokens, clamped to the embedder's window. "
        "Default: settings.chunk_size (512).",
    )
    p_ingest.add_argument(
        "--vision",
        action="store_true",
        help="Also read figures/charts into text chunks via a vision model "
        "(default Claude Sonnet; set ACCURAG_VISION_MODEL to override). "
        "PDFs only; extra parse + per-figure API cost.",
    )
    _add_embedder_flag(p_ingest)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Answer a query with grounded citations.")
    p_ask.add_argument("query", help="The question to ask.")
    p_ask.add_argument(
        "--strategy",
        default="dense",
        choices=["dense", "hybrid", "hybrid_rerank"],
        help="Retrieval strategy.",
    )
    p_ask.add_argument("-k", type=int, default=5, help="Number of chunks to retrieve.")
    _add_embedder_flag(p_ask)
    p_ask.add_argument(
        "--llm",
        choices=["anthropic", "openai"],
        default=None,
        help="Answer LLM. Default = Anthropic Claude; use 'openai' to run with "
        "an OpenAI chat key instead.",
    )
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("evaluate", help="Score strategies over a golden set.")
    p_eval.add_argument("golden", help="Path to the golden-set JSON file.")
    p_eval.add_argument("-k", type=int, default=5, help="Number of chunks to retrieve.")
    p_eval.add_argument(
        "--strategies",
        default="dense,hybrid,hybrid_rerank",
        help="Comma-separated strategies to score. Use 'dense,hybrid' to stay "
        "fully offline (hybrid_rerank needs a Cohere key).",
    )
    p_eval.add_argument(
        "--answer-quality",
        action="store_true",
        help="Also grade faithfulness + answer relevancy with an LLM judge "
        "(generates answers + judge calls, costs API; judge=gpt-4o-mini).",
    )
    _add_embedder_flag(p_eval)
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    import logging

    # Surface pipeline progress (per-doc ingest lines, etc.) on the console.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
