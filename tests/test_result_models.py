from accurag.models import (
    Answer,
    Chunk,
    EvalReport,
    EvalRow,
    GoldenQA,
    RetrievedChunk,
)


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=1,
        text="some text",
        source_title="A Paper",
        theme="rag",
        section=None,
        url="https://example.com",
    )


def test_retrieved_chunk_holds_chunk_score_rank():
    rc = RetrievedChunk(chunk=_chunk("1-0"), score=0.42, rank=0)
    assert rc.chunk.chunk_id == "1-0"
    assert rc.score == 0.42
    assert rc.rank == 0


def test_answer_carries_typed_sources():
    """sources are RetrievedChunk objects: typed provenance, no parsing needed."""
    ans = Answer(
        text="The answer.",
        sources=[RetrievedChunk(chunk=_chunk("1-0"), score=0.9, rank=0)],
    )
    assert ans.text == "The answer."
    assert ans.sources[0].chunk.chunk_id == "1-0"
    assert ans.sources[0].chunk.url == "https://example.com"
    assert ans.sources[0].score == 0.9


def test_answer_sources_default_empty():
    ans = Answer(text="I don't know.")
    assert ans.sources == []


def test_golden_qa_fields():
    qa = GoldenQA(
        question="What is RAG?",
        ground_truth="Retrieval-augmented generation.",
        relevant_chunk_ids=["1-0", "2-3"],
    )
    assert qa.question == "What is RAG?"
    assert qa.relevant_chunk_ids == ["1-0", "2-3"]


def test_eval_report_to_markdown_has_header_and_strategies():
    report = EvalReport(
        rows=[
            EvalRow(
                strategy="dense",
                context_precision=0.80,
                context_recall=0.70,
                faithfulness=0.90,
                answer_relevancy=0.85,
                recall_at_k=0.75,
                mrr=0.60,
                p95_latency_ms=1234.5,
            ),
            EvalRow(
                strategy="hybrid+rerank",
                context_precision=None,
                context_recall=None,
                faithfulness=None,
                answer_relevancy=None,
                recall_at_k=None,
                mrr=None,
                p95_latency_ms=None,
            ),
        ]
    )
    md = report.to_markdown()
    # header row
    assert "Config" in md
    assert "Ctx Precision" in md
    assert "Ctx Recall" in md
    assert "Faithfulness" in md
    assert "Answer Rel." in md
    assert "Recall@k" in md
    assert "MRR" in md
    assert "P95 Latency(ms)" in md
    # strategy names present
    assert "dense" in md
    assert "hybrid+rerank" in md
    # None rendered as em dash
    assert "—" in md
    # full-coverage runs don't clutter the table with a coverage column
    assert "Judge Cov." not in md


def test_eval_report_shows_judge_coverage_only_when_partial():
    """The Judge Cov. column appears only when a run was partial (< 1.0), so the
    table stays clean normally and stays accurate when the judge skipped samples."""
    full = EvalReport(rows=[EvalRow(strategy="dense", faithfulness=0.9, judge_coverage=1.0)])
    assert "Judge Cov." not in full.to_markdown()

    partial = EvalReport(rows=[EvalRow(strategy="dense", faithfulness=0.9, judge_coverage=0.8)])
    md = partial.to_markdown()
    assert "Judge Cov." in md
    assert "0.800" in md
