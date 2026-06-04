"""Tests for fetch.py — all hermetic, no network."""

from accurag.models import ManifestEntry

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _entry(**kwargs) -> ManifestEntry:
    """Build a minimal valid ManifestEntry, override with kwargs."""
    defaults = {
        "id": 1,
        "title": "Test Paper",
        "authors": "Author et al.",
        "year": 2024,
        "arxiv_id": None,
        "doi": None,
        "pdf_url": "https://arxiv.org/pdf/2005.11401",
        "source": "arxiv",
        "theme": "rag_foundations",
        "has_tables_or_figures": False,
        "verified": True,
        "note": "",
    }
    defaults.update(kwargs)
    return ManifestEntry(**defaults)


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_returns_list_of_manifest_entries(tmp_path):
    import json

    from accurag.fetch import load_manifest

    data = {
        "documents": [
            {
                "id": 1,
                "title": "Test Paper",
                "authors": "A et al.",
                "year": 2024,
                "arxiv_id": "2005.11401",
                "doi": None,
                "pdf_url": "https://arxiv.org/pdf/2005.11401",
                "source": "arxiv",
                "theme": "rag_foundations",
                "has_tables_or_figures": True,
                "verified": True,
                "note": "",
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(data))

    entries = load_manifest(manifest_path)
    assert len(entries) == 1
    assert isinstance(entries[0], ManifestEntry)
    assert entries[0].id == 1
    assert entries[0].arxiv_id == "2005.11401"


# ---------------------------------------------------------------------------
# target_path — the three cases
# ---------------------------------------------------------------------------


class TestTargetPath:
    def test_arxiv_source_gives_pdf(self, tmp_path):
        from accurag.fetch import target_path

        entry = _entry(id=1, source="arxiv", pdf_url="https://arxiv.org/pdf/2005.11401")
        p = target_path(entry, tmp_path)
        assert p == tmp_path / "1.pdf"

    def test_vendor_pdf_url_gives_pdf(self, tmp_path):
        from accurag.fetch import target_path

        entry = _entry(
            id=41,
            source="vendor",
            pdf_url="https://assets.anthropic.com/m/abc/original/Contextual-Retrieval-Appendix-2.pdf",
        )
        p = target_path(entry, tmp_path)
        assert p == tmp_path / "41.pdf"

    def test_vendor_html_url_gives_html(self, tmp_path):
        from accurag.fetch import target_path

        entry = _entry(
            id=26,
            source="vendor",
            pdf_url="https://research.trychroma.com/evaluating-chunking",
        )
        p = target_path(entry, tmp_path)
        assert p == tmp_path / "26.html"

    def test_other_source_pdf_url_gives_pdf(self, tmp_path):
        from accurag.fetch import target_path

        entry = _entry(
            id=12,
            source="other",
            pdf_url="https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf",
        )
        p = target_path(entry, tmp_path)
        assert p == tmp_path / "12.pdf"

    def test_vendor_html_source_gives_html(self, tmp_path):
        from accurag.fetch import target_path

        # Vendor page with no .pdf extension
        entry = _entry(
            id=44,
            source="vendor",
            pdf_url="https://cohere.com/blog/rerank-3",
        )
        p = target_path(entry, tmp_path)
        assert p == tmp_path / "44.html"


# ---------------------------------------------------------------------------
# fetch_all — no network; verify skip-if-exists and continue-on-error logic
# ---------------------------------------------------------------------------


class TestFetchAll:
    def test_skips_already_downloaded_file(self, tmp_path, monkeypatch):
        """If the target file already exists, fetch_all must not call httpx."""
        from accurag.fetch import fetch_all

        entry = _entry(id=1, source="arxiv", pdf_url="https://arxiv.org/pdf/2005.11401")
        # Pre-create the target so it's "already downloaded"
        existing = tmp_path / "1.pdf"
        existing.write_bytes(b"already here")

        # Patch httpx.Client so any real network call would raise
        import accurag.fetch as fetch_mod

        class NeverCalledClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, *args, **kwargs):
                raise AssertionError("httpx.Client.get should not be called for an existing file")

        monkeypatch.setattr(fetch_mod.httpx, "Client", lambda **kwargs: NeverCalledClient())

        paths = fetch_all([entry], tmp_path)
        assert paths == [existing]

    def test_continues_after_bad_url(self, tmp_path, monkeypatch):
        """A per-URL failure must not abort the rest of the run."""
        from accurag.fetch import fetch_all

        bad_entry = _entry(id=99, source="arxiv", pdf_url="https://arxiv.org/pdf/DOES_NOT_EXIST")
        good_entry = _entry(id=1, source="arxiv", pdf_url="https://arxiv.org/pdf/2005.11401")

        import httpx

        import accurag.fetch as fetch_mod

        call_count = {"n": 0}

        class FakeResponse:
            content = b"fake pdf content"

            def raise_for_status(self):
                pass

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, url, **kwargs):
                call_count["n"] += 1
                if "DOES_NOT_EXIST" in url:
                    raise httpx.HTTPError("connection refused")
                return FakeResponse()

        monkeypatch.setattr(fetch_mod.httpx, "Client", lambda **kwargs: FakeClient())

        paths = fetch_all([bad_entry, good_entry], tmp_path)
        # Only the good entry should be in the result
        assert len(paths) == 1
        assert paths[0] == tmp_path / "1.pdf"
        assert (tmp_path / "1.pdf").read_bytes() == b"fake pdf content"
        # The bad entry should NOT have created a file
        assert not (tmp_path / "99.pdf").exists()

    def test_returns_paths_for_all_successful(self, tmp_path, monkeypatch):
        """fetch_all returns one Path per successfully fetched (or pre-existing) entry."""
        from accurag.fetch import fetch_all

        entries = [
            _entry(id=1, source="arxiv", pdf_url="https://arxiv.org/pdf/AAA"),
            _entry(id=2, source="vendor", pdf_url="https://example.com/report"),
        ]

        import accurag.fetch as fetch_mod

        class FakeResponse:
            content = b"data"

            def raise_for_status(self):
                pass

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def get(self, url, **kwargs):
                return FakeResponse()

        monkeypatch.setattr(fetch_mod.httpx, "Client", lambda **kwargs: FakeClient())

        paths = fetch_all(entries, tmp_path)
        assert len(paths) == 2
        assert tmp_path / "1.pdf" in paths
        assert tmp_path / "2.html" in paths
