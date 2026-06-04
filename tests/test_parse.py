"""Tests for accurag.parse. Trivial import/callable check only (does not touch the network or parse from disk)."""

import inspect


def test_parse_file_is_importable_and_callable() -> None:
    from accurag.parse import parse_file

    assert callable(parse_file)


def test_parse_file_accepts_path_argument() -> None:
    from accurag.parse import parse_file

    sig = inspect.signature(parse_file)
    params = list(sig.parameters.keys())
    assert "path" in params
