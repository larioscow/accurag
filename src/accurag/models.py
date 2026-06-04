from typing import Literal

from pydantic import BaseModel


class ManifestEntry(BaseModel):
    """One row of corpus/manifest.json: a single source document."""

    id: int
    title: str
    authors: str
    year: int
    arxiv_id: str | None = None
    doi: str | None = None
    pdf_url: str
    source: Literal["arxiv", "vendor", "other"]
    theme: str
    has_tables_or_figures: bool
    verified: bool
    note: str = ""


class Chunk(BaseModel):
    """One retrievable piece of a document. The fields let a citation be shown
    and verified, and let chunks be filtered by document or theme."""

    chunk_id: str  # f"{doc_id}-{ordinal}" (the [x:y] citation form)
    doc_id: int  # which ManifestEntry it came from
    text: str  # the content itself
    source_title: str  # human-readable citation ("Lost in the Middle")
    theme: str  # enables filtering ("only agentic-patterns papers")
    section: str | None  # which heading the chunk came from
    url: str  # so a human can open it and verify


class RetrievedChunk(BaseModel):
    """A Chunk returned by retrieval, with its relevance score and ordering."""

    chunk: Chunk
    score: float
    rank: int


class Answer(BaseModel):
    """A grounded answer plus the sources it was generated from.

    ``sources`` are the retrieved chunks the answer was conditioned on. They are
    known in Python from retrieval rather than parsed out of the model's output,
    so a consumer gets typed provenance (``source.chunk.source_title``, ``.url``,
    ``.section``, ``.text``, ``source.score``, ``.rank``) to render source cards
    or links without parsing anything out of ``text``.
    """

    text: str
    sources: list[RetrievedChunk] = []


class GoldenQA(BaseModel):
    """One hand-built evaluation example: a question, its ground-truth
    answer, and the chunk_ids that should be retrieved to answer it."""

    question: str
    ground_truth: str
    relevant_chunk_ids: list[str]


class EvalRow(BaseModel):
    """Scores for one retrieval/answer strategy in the comparison table."""

    strategy: str
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    recall_at_k: float | None = None
    mrr: float | None = None
    p95_latency_ms: float | None = None
    # Fraction of answer-quality samples the judge actually graded (< 1.0 means
    # some judge calls failed and were skipped, so the faithfulness / answer-rel
    # means rest on fewer judgements). Surfaced in the table only when < 1.0.
    judge_coverage: float | None = None


class EvalReport(BaseModel):
    """The naive/hybrid/rerank comparison table."""

    rows: list[EvalRow]

    def to_markdown(self) -> str:
        """Render the rows as a GitHub-flavored markdown table.
        None scores show as an em dash. A "Judge Cov." column is added only when
        some run was partial (coverage < 1.0), so the column appears when the
        judge skipped some samples and is omitted when it graded everything."""
        show_coverage = any(
            r.judge_coverage is not None and r.judge_coverage < 1.0 for r in self.rows
        )
        headers = [
            "Config",
            "Ctx Precision",
            "Ctx Recall",
            "Faithfulness",
            "Answer Rel.",
            "Recall@k",
            "MRR",
            "P95 Latency(ms)",
        ]
        if show_coverage:
            headers.append("Judge Cov.")

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.3f}"

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in self.rows:
            cells = [
                row.strategy,
                fmt(row.context_precision),
                fmt(row.context_recall),
                fmt(row.faithfulness),
                fmt(row.answer_relevancy),
                fmt(row.recall_at_k),
                fmt(row.mrr),
                fmt(row.p95_latency_ms),
            ]
            if show_coverage:
                cells.append(fmt(row.judge_coverage))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
