"""Corpus manifest loading and document fetching.

Rules
-----
- target_path: ``<id>.pdf`` when ``entry.source == "arxiv"`` or
  ``entry.pdf_url`` ends with ``.pdf``; otherwise ``<id>.html``.
- fetch_all: skips files that already exist on disk; catches per-URL
  exceptions (logs + continues) so one bad URL never aborts the run.
- httpx is imported at module level (it is a pure networking lib, not a
  heavy SDK with API-key requirements — no lazy-import needed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from accurag.models import ManifestEntry

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> list[ManifestEntry]:
    """Parse ``path`` (a JSON file with a ``"documents"`` list) and return
    a list of :class:`~accurag.models.ManifestEntry` objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ManifestEntry(**doc) for doc in raw["documents"]]


def target_path(entry: ManifestEntry, raw_dir: Path) -> Path:
    """Return the local destination path for *entry*.

    Naming convention
    -----------------
    ``<id>.pdf``  — when ``entry.source == "arxiv"`` **or** ``entry.pdf_url``
                    ends with ``.pdf`` (case-insensitive).
    ``<id>.html`` — all other cases (vendor/other HTML pages).
    """
    is_pdf = entry.source == "arxiv" or entry.pdf_url.lower().endswith(".pdf")
    ext = "pdf" if is_pdf else "html"
    return Path(raw_dir) / f"{entry.id}.{ext}"


def fetch_all(
    entries: list[ManifestEntry],
    raw_dir: Path,
    *,
    timeout: float = 60.0,
) -> list[Path]:
    """Download all *entries* into *raw_dir*; return paths of successful files.

    Behaviour
    ---------
    - Already-downloaded files are skipped (idempotent).
    - A single failed URL is logged and skipped; the rest of the run continues.
    - Uses a browser-like User-Agent and follows redirects automatically.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _BROWSER_UA},
    ) as client:
        for entry in entries:
            dest = target_path(entry, raw_dir)

            if dest.exists():
                logger.info("skip (exists): %s", dest)
                results.append(dest)
                continue

            try:
                logger.info("fetch %s -> %s", entry.pdf_url, dest)
                response = client.get(entry.pdf_url)
                response.raise_for_status()
                dest.write_bytes(response.content)
                results.append(dest)
                logger.info("saved %d bytes -> %s", len(response.content), dest)
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to fetch %s: %s", entry.pdf_url, exc)

    return results
