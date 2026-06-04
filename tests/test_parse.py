"""Tests for accurag.parse — trivial import/callable check only (no network, no disk parse)."""

import inspect


def test_parse_file_is_importable_and_callable() -> None:
    from accurag.parse import parse_file

    assert callable(parse_file)


def test_parse_file_accepts_path_argument() -> None:
    from accurag.parse import parse_file

    sig = inspect.signature(parse_file)
    params = list(sig.parameters.keys())
    assert "path" in params
