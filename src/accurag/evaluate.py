"""Evaluation metrics for the retrieval/answer comparison table.

Two kinds of metric live here:

- Pure retrieval metrics (``recall_at_k``, ``mrr``) use no external libraries
  and no network, and are unit-tested on hand-computed examples. They score
  whether the right chunks were retrieved, independent of any LLM.
- Answer-quality metrics (``ragas_answer_scores``) cover faithfulness, answer
  relevancy, and context precision/recall, which require an LLM judge. Ragas is
  imported lazily inside that function and the judge LLM/embeddings are
  injectable, so importing this module (and running the unit tests) needs no
  API key, no network, and no ``ragas`` install.

``build_report`` assembles per-strategy rows into the ``EvalReport`` whose
``to_markdown()`` renders the project's main artifact, the
naive-vs-hybrid-vs-rerank comparison table.
"""

from __future__ import annotations

import logging
from typing import Any

from accurag.models import EvalReport, EvalRow

logger = logging.getLogger(__name__)


def recall_at_k(
    relevant_ids: set[str] | list[str],
    retrieved_ids: list[str],
) -> float:
    """Fraction of the relevant chunk_ids that appear in ``retrieved_ids``.

    This is recall over whatever ``k`` chunks the caller chose to retrieve, so
    pass the top-k retrieved ids in. Duplicate retrieved ids are de-duplicated
    so they cannot inflate the score.

    Args:
        relevant_ids:  The chunk_ids that should be retrieved (the golden set).
        retrieved_ids: The chunk_ids actually retrieved, best-first.

    Returns:
        ``hits / len(relevant)`` in ``[0, 1]``. An empty ``relevant_ids`` means
        there is nothing to find, so recall is trivially ``1.0``.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 1.0
    retrieved = set(retrieved_ids)
    hits = len(relevant & retrieved)
    return hits / len(relevant)


def mrr(
    relevant_ids: set[str] | list[str],
    ranked_ids: list[str],
) -> float:
    """Reciprocal rank of the first relevant chunk in ``ranked_ids``.

    Args:
        relevant_ids: The chunk_ids that count as relevant.
        ranked_ids:   The retrieved chunk_ids in rank order (best first).

    Returns:
        ``1 / rank`` of the earliest relevant id (rank is 1-based), or ``0.0``
        if no relevant id appears or there are no relevant ids at all.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


# ---------------------------------------------------------------------------
# Answer-quality metrics via a lightweight LLM judge.
#
# This path avoids Ragas: Ragas pulls the langchain ecosystem and (in the
# versions we hit) breaks on a missing langchain_community vertexai import. A
# small judge over the OpenAI SDK we already depend on avoids that dependency,
# keeps the library lean, and stays inspectable. The judge model (gpt-4o-mini)
# differs from the answer generator (Claude), so the judge is not grading its
# own output.
# ---------------------------------------------------------------------------

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "faithfulness": {
            "type": "number",
            "description": (
                "0..1: the fraction of the ANSWER's factual claims that are "
                "directly supported by the CONTEXT. 1.0 = every claim grounded; "
                "0.0 = unsupported/hallucinated. If the answer makes no factual "
                "claims (e.g. 'I don't know'), return 1.0, since nothing is unfaithful."
            ),
        },
        "answer_relevancy": {
            "type": "number",
            "description": (
                "0..1: how directly and completely the ANSWER addresses the "
                "QUESTION. A refusal like 'I don't know' is NOT relevant, so score "
                "it near 0.0. A complete, on-topic answer scores near 1.0."
            ),
        },
    },
    "required": ["faithfulness", "answer_relevancy"],
}


def make_judge_client() -> Any:
    """Build an OpenAI client for the judge from settings (lazy import)."""
    from openai import OpenAI

    from accurag.config import settings

    return OpenAI(api_key=settings.openai_api_key or None)


def judge_answer_quality(
    samples: list[dict[str, Any]],
    client: Any,
    model: str = "gpt-4o-mini",
) -> dict[str, float | None]:
    """LLM-judge faithfulness and answer relevancy, averaged over the samples.

    Args:
        samples: list of ``{"question": str, "answer": str, "contexts": list[str]}``.
        client:  an OpenAI-compatible client (injectable for tests) exposing
                 ``chat.completions.create`` with tool calling.
        model:   the judge model (must differ from the answer generator).

    A single bad judge reply (refusal, empty tool call, truncated output, or a
    transient API error) skips that sample and is counted rather than aborting
    the run. Means are computed over the graded samples.

    Returns:
        ``{"faithfulness": mean|None, "answer_relevancy": mean|None,
        "coverage": graded/total|None}``. A ``coverage`` below 1.0 means some
        samples were skipped, so the means rest on fewer judgements.
    """
    import json

    tool = {
        "type": "function",
        "function": {
            "name": "grade",
            "description": "Grade a RAG answer's faithfulness and relevancy.",
            "parameters": _JUDGE_SCHEMA,
            "strict": True,
        },
    }
    faith: list[float] = []
    rel: list[float] = []
    failures = 0
    for s in samples:
        ctx = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(s["contexts"]))
        prompt = (
            "Grade the RAG answer below.\n\n"
            f"QUESTION:\n{s['question']}\n\n"
            f"CONTEXT:\n{ctx or '(no context retrieved)'}\n\n"
            f"ANSWER:\n{s['answer']}\n\n"
            "Score faithfulness (claims supported by CONTEXT) and answer_relevancy "
            "(does it answer the QUESTION)."
        )
        # A single bad judge response (a refusal, a missing tool call,
        # malformed/truncated output, or a transient API error) should not kill
        # the whole eval. Skip the sample and report coverage. (Strict structured
        # output already removes the malformed-JSON class; this guards the rest.)
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=200,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "grade"}},
                messages=[{"role": "user", "content": prompt}],
            )
            out = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
            faith.append(float(out["faithfulness"]))
            rel.append(float(out["answer_relevancy"]))
        except Exception as exc:  # noqa: BLE001 (judge quality is best-effort, never fatal)
            failures += 1
            logger.warning("[eval] judge failed on a sample (%s), skipping", exc)

    graded = len(faith)
    if failures:
        logger.warning(
            "[eval] answer-quality judged %d/%d samples (%d skipped)",
            graded,
            len(samples),
            failures,
        )
    return {
        "faithfulness": sum(faith) / graded if graded else None,
        "answer_relevancy": sum(rel) / graded if graded else None,
        "coverage": graded / len(samples) if samples else None,
    }


def build_report(rows: list[dict[str, Any]] | list[EvalRow]) -> EvalReport:
    """Assemble per-strategy ``rows`` into an ``EvalReport``.

    Accepts either ready-made ``EvalRow`` instances or plain dicts (which are
    validated into ``EvalRow``), so callers can build rows however is most
    convenient. Order is preserved.
    """
    eval_rows = [row if isinstance(row, EvalRow) else EvalRow(**row) for row in rows]
    return EvalReport(rows=eval_rows)


def ragas_answer_scores(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: list[str],
    *,
    llm: Any,
    embeddings: Any,
) -> dict[str, float | None]:
    """Score answer quality with Ragas (faithfulness, answer relevancy,
    context precision, context recall) averaged over the samples.

    Ragas and its judge LLM are heavy and require keys, so:
    - ``ragas`` is imported lazily inside this function (the module imports and
      the unit tests run without it installed);
    - the judge ``llm`` and ``embeddings`` are injected by the caller. Wrap your
      provider once (e.g. ``LangchainLLMWrapper(ChatOpenAI(...))``) and pass it
      in. This keeps provider choice out of the metric code.

    Args:
        questions:  One user question per sample.
        answers:    The generated answer for each question.
        contexts:   The retrieved context strings backing each answer.
        references: The ground-truth answer for each question.
        llm:        A Ragas-compatible wrapped judge LLM.
        embeddings: A Ragas-compatible wrapped embeddings model (needed by
                    ResponseRelevancy).

    Returns:
        A mapping from metric name to its mean score over the dataset.
    """
    # Lazy imports: these pull in ragas and its deps, which the unit tests avoid.
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )

    samples = [
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": context,
            "reference": reference,
        }
        for question, answer, context, reference in zip(
            questions, answers, contexts, references, strict=True
        )
    ]
    dataset = EvaluationDataset.from_list(samples)

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
    return _mean_scores(result)


def _mean_scores(result: Any) -> dict[str, float | None]:
    """Reduce a Ragas ``EvaluationResult`` to per-metric mean scores.

    Uses the result's pandas view so we are agnostic to per-version dict keys.
    """
    frame = result.to_pandas()
    scores: dict[str, float | None] = {}
    for column in frame.columns:
        series = frame[column]
        numeric = series.dropna()
        if numeric.empty or not _is_numeric(numeric.iloc[0]):
            continue
        scores[column] = float(numeric.mean())
    return scores


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
