"""Rate-limit retry behaviour for rerank (hermetic, no network, no real sleep)."""

from __future__ import annotations

from accurag.rerank import _is_rate_limit, _rerank_with_retry


class _TooManyRequestsError(Exception):
    pass


def test_is_rate_limit_detection():
    assert _is_rate_limit(_TooManyRequestsError("nope"))  # by type name
    assert _is_rate_limit(Exception("status_code: 429"))  # by 429 in message
    assert _is_rate_limit(Exception("rate limit exceeded"))  # by phrase
    assert not _is_rate_limit(Exception("connection reset"))


class _FakeV2:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def rerank(self, **_):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _TooManyRequestsError("429")
        return "OK"


class _FakeClient:
    def __init__(self, fail_times: int):
        self.v2 = _FakeV2(fail_times)


def test_rerank_retries_then_succeeds():
    client = _FakeClient(fail_times=2)
    out = _rerank_with_retry(client, "m", "q", ["a"], 1, wait_seconds=0)
    assert out == "OK"
    assert client.v2.calls == 3  # 2 failures + 1 success


def test_rerank_gives_up_after_max_retries():
    client = _FakeClient(fail_times=99)
    try:
        _rerank_with_retry(client, "m", "q", ["a"], 1, max_retries=3, wait_seconds=0)
        raise AssertionError("expected the rate-limit error to propagate")
    except _TooManyRequestsError:
        pass
    assert client.v2.calls == 3


def test_non_rate_limit_error_propagates_immediately():
    class _C:
        class v2:
            @staticmethod
            def rerank(**_):
                raise ValueError("bad request")

    try:
        _rerank_with_retry(_C(), "m", "q", ["a"], 1, wait_seconds=0)
        raise AssertionError("expected ValueError to propagate")
    except ValueError:
        pass
