from accurag.evaluate import build_report, mrr, recall_at_k
from accurag.models import EvalReport, EvalRow

# --- recall_at_k -------------------------------------------------------------


def test_recall_at_k_all_relevant_retrieved():
    # both relevant ids are present in the retrieved list -> 1.0
    assert recall_at_k({"a", "b"}, ["a", "x", "b"]) == 1.0


def test_recall_at_k_half_retrieved():
    # 1 of 2 relevant ids retrieved -> 0.5
    assert recall_at_k(["a", "b"], ["a", "x", "y"]) == 0.5


def test_recall_at_k_none_retrieved():
    assert recall_at_k({"a", "b"}, ["x", "y"]) == 0.0


def test_recall_at_k_empty_relevant_is_one():
    # nothing to find -> trivially perfect recall
    assert recall_at_k(set(), ["x", "y"]) == 1.0


def test_recall_at_k_dedupes_retrieved():
    # duplicate retrieved ids must not inflate the count
    assert recall_at_k({"a"}, ["a", "a", "a"]) == 1.0


# --- mrr ---------------------------------------------------------------------


def test_mrr_first_position():
    # first relevant id is at rank 1 -> 1/1
    assert mrr({"a", "b"}, ["a", "x", "y"]) == 1.0


def test_mrr_third_position():
    # first relevant id appears at rank 3 -> 1/3
    assert mrr({"c"}, ["a", "b", "c"]) == 1.0 / 3.0


def test_mrr_no_relevant_found():
    assert mrr({"z"}, ["a", "b", "c"]) == 0.0


def test_mrr_empty_relevant_is_zero():
    assert mrr(set(), ["a", "b"]) == 0.0


def test_mrr_uses_earliest_relevant():
    # second item is relevant -> 1/2, even though a later one is too
    assert mrr(["b", "c"], ["a", "b", "c"]) == 0.5


# --- build_report ------------------------------------------------------------


def test_build_report_from_dicts():
    rows = [
        {"strategy": "dense", "recall_at_k": 0.5, "mrr": 0.75},
        {"strategy": "hybrid", "recall_at_k": 0.8, "mrr": 0.9},
    ]
    report = build_report(rows)
    assert isinstance(report, EvalReport)
    assert [r.strategy for r in report.rows] == ["dense", "hybrid"]
    md = report.to_markdown()
    assert "dense" in md
    assert "hybrid" in md


def test_build_report_from_evalrows():
    rows = [EvalRow(strategy="rerank", recall_at_k=0.9, mrr=0.95)]
    report = build_report(rows)
    assert isinstance(report, EvalReport)
    assert report.rows[0].strategy == "rerank"
    assert "rerank" in report.to_markdown()


# --- custom LLM-judge answer-quality metrics (hermetic, fake client) ---
def test_judge_answer_quality_averages_scores():
    import json
    from types import SimpleNamespace

    from accurag.evaluate import judge_answer_quality

    def _resp(f, r):
        tc = SimpleNamespace(
            function=SimpleNamespace(
                arguments=json.dumps({"faithfulness": f, "answer_relevancy": r})
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]))])

    class _Completions:
        def __init__(self, scores):
            self.scores = scores
            self.i = 0

        def create(self, **_):
            f, r = self.scores[self.i]
            self.i += 1
            return _resp(f, r)

    class _Client:
        def __init__(self, scores):
            self.chat = SimpleNamespace(completions=_Completions(scores))

    client = _Client([(1.0, 0.0), (0.5, 1.0)])
    samples = [{"question": "q", "answer": "a", "contexts": ["c"]}] * 2
    out = judge_answer_quality(samples, client=client)
    assert out["faithfulness"] == 0.75
    assert out["answer_relevancy"] == 0.5


def test_judge_answer_quality_empty_samples():
    from accurag.evaluate import judge_answer_quality

    out = judge_answer_quality([], client=None)
    assert out["faithfulness"] is None and out["answer_relevancy"] is None


def test_judge_answer_quality_survives_bad_judge_responses():
    """A bad judge reply (API error, or a malformed no-tool-call response) must
    not kill the run. The sample is skipped, the good ones are still scored, and
    coverage reflects the loss."""
    import json
    from types import SimpleNamespace

    from accurag.evaluate import judge_answer_quality

    def _good(f, r):
        tc = SimpleNamespace(
            function=SimpleNamespace(
                arguments=json.dumps({"faithfulness": f, "answer_relevancy": r})
            )
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]))])

    class _FlakyClient:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **_):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("judge API blew up")  # transient API/network error
            if self.calls == 2:
                # malformed: no tool_calls -> tool_calls[0] would crash the naive parser
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
                )
            return _good(1.0, 0.5)  # the one good sample

    samples = [{"question": "q", "answer": "a", "contexts": ["c"]}] * 3
    out = judge_answer_quality(samples, client=_FlakyClient())  # must not raise
    # only the third sample graded; means rest on it, coverage shows 1/3
    assert out["faithfulness"] == 1.0
    assert out["answer_relevancy"] == 0.5
    assert out["coverage"] == 1 / 3
