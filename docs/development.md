# Development

How to set up the repo, run the exact gate CI runs, and build the docs. The test suite is hermetic — no API keys, no network — so you can clone and run it cold.

## Setup

Clone and install the dev tooling. Use either `uv` or plain pip:

```bash
git clone <this repo> && cd accurag

# uv (what CI uses)
uv sync --extra dev

# or pip
pip install -e ".[dev]"
```

The `dev` extra brings `pytest`, `ruff`, and `mypy`. That is enough to run the tests. CI additionally installs the `ingest` extra so mypy type-checks against the real Docling / PyMuPDF / tiktoken API (see below).

## The quality gate

CI (`.github/workflows/ci.yml`) runs four checks, in order. Run them locally exactly as CI does:

```bash
ruff check src tests          # lint
ruff format --check src tests # formatting
mypy                          # types (checks src/accurag)
pytest -q                     # 140 tests
```

With `uv`, prefix each with `uv run` (e.g. `uv run ruff check src tests`).

!!! note "What CI installs"
    CI runs `uv sync --extra dev --extra ingest`. The `ingest` extra is included **only** so mypy type-checks against the real `docling` / `pymupdf` / `tiktoken` API — the test suite itself is fully hermetic and needs no keys, no network, and no parser SDKs. Everything else in the workflow is the four commands above.

All four are green on the tree: ruff- and mypy-clean, 140 passing tests.

## Conventions

A few project conventions worth knowing before you change code:

- **Lazy imports keep `import accurag` key-free.** Heavy and optional SDKs (parsers, eval, observability) are imported inside the functions that use them, not at module top level, so importing the package is cheap and pulls no provider keys. This is the architecture, not an oversight — `PLC0415` (import-outside-top-level) is **ignored by design** in `pyproject.toml`. New code that touches an optional SDK should follow the same pattern.
- **Tests use fakes, not the network.** The suite mirrors SDK signatures with in-function fakes and runs Qdrant in-memory, so it is hermetic. Tests are also allowed private access and toy values by design (`tests/**` has its own ruff per-file ignores).
- **Small, single-responsibility modules.** The pipeline is built from small typed functions under `src/accurag/`; `pipeline.py` is the only file that wires them together. Keep functions small and typed when adding to it.

## Building the docs locally

The docs site is Material for MkDocs. It needs the `docs` extra:

```bash
pip install -e ".[docs]"   # or: uv sync --extra docs
mkdocs serve
```

`mkdocs serve` runs a live-reloading preview at `http://127.0.0.1:8000`. See [getting started](getting-started.md) to use the library, or the [API reference](reference/accurag/index.md) for the generated surface.
