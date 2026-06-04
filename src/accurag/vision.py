"""Vision figure extraction — read charts/figures into text at index time.

Text parsers can't read the content of figures (numbers in bar charts, axes,
diagram relationships), so figure-heavy documents lose information the retriever
can never match. This module renders each detected figure to an image and has a
vision model describe/extract it as plain text, producing extra chunks the
retriever *can* match.

It is an OPTIONAL, index-time step (`RagPipeline.ingest(vision=True)`):
- Docling picture rendering is imported lazily (needs `accurag[ingest]`).
- The vision model is configurable (`settings.vision_model`); it defaults to
  **Claude Sonnet**, which reads multi-panel chart *structure* (titles, axes,
  series, labels) that gpt-4o-mini misreads. Set `model="gpt-4o-mini"` for a
  cheaper bulk pass on simple figures.

Honest limitation (measured — see docs/EVAL_RESULTS.md): vision reads chart
*structure* reliably but *guesses bar/line heights*, so extracted **numeric
values are unreliable**. A quantitative chart answer can come back grounded in a
real source chunk yet numerically wrong, and deterministic provenance does not
catch it (the chunk exists; its number was mis-read). Treat this as a best-effort
recall aid for figure-heavy docs, NOT a trustworthy table/figure extractor.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

from accurag.models import Chunk, ManifestEntry

_FIG_PROMPT = (
    "This is a figure from a technical paper. Extract everything a reader needs "
    "from it as plain text: title/caption, axis labels, every series, and all "
    "numeric values you can read (including approximate values for bar/line "
    "heights). If it's a diagram, describe its components and their relationships. "
    "Be concrete and exhaustive — this text replaces the image for search."
)

_converter = None


def _figure_converter():
    """Docling converter with picture-image rendering on (CPU-pinned, OCR off)."""
    global _converter  # noqa: PLW0603
    if _converter is None:
        try:
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Parser deps not installed — run: pip install 'accurag[ingest]'"
            ) from exc

        opts = PdfPipelineOptions()
        opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=4)
        opts.do_ocr = False
        opts.generate_picture_images = True  # the whole point — render figures
        opts.images_scale = 2.0  # higher res so chart text/values are legible
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts, backend=PyPdfiumDocumentBackend
                )
            }
        )
    return _converter


def extract_figure_images(path: Path) -> list[Any]:
    """Return a PIL image for each figure Docling detects in the PDF."""
    doc = _figure_converter().convert(str(path)).document
    images = []
    for pic in doc.pictures:
        img = pic.get_image(doc)
        if img is not None:
            images.append(img)
    return images


def _b64_png(image: Any) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _vision_client(model: str, client: Any) -> Any:
    """Build the right provider client for *model* (Anthropic for claude-*, else OpenAI)."""
    if client is not None:
        return client
    from accurag.config import settings

    if model.startswith("claude"):
        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key or None)


def describe_figure(image: Any, client: Any = None, model: str | None = None) -> str:
    """Describe/extract one figure image as plain text via a vision model.

    Defaults to ``settings.vision_model`` (Claude Sonnet — it reads multi-panel
    chart *structure* that gpt-4o-mini misreads). Pass ``model="gpt-4o-mini"``
    for a cheaper bulk pass on simple figures.

    Reliability: structure (titles/axes/series) yes; **numeric values no** — the
    model guesses bar/line heights, so quantitative extractions are unreliable
    (see the module docstring and docs/EVAL_RESULTS.md).
    """
    from accurag.config import settings

    model = model or settings.vision_model
    client = _vision_client(model, client)
    b64 = _b64_png(image)

    if model.startswith("claude"):
        resp = client.messages.create(
            model=model,
            max_tokens=700,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _FIG_PROMPT},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                    ],
                }
            ],
        )
        # Filter to text blocks — a leading thinking/redacted block has no .text
        # (mirrors AnthropicLLM.answer). Missing/empty text -> "" (caller skips it).
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    resp = client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _FIG_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    )
    # content is None on a refusal / max_tokens truncation -> "" (caller skips it).
    return (resp.choices[0].message.content or "").strip()


def figure_chunks(
    path: Path,
    entry: ManifestEntry,
    client: Any = None,
    model: str | None = None,
) -> list[Chunk]:
    """Extract each figure from *path* and return one text Chunk per figure.

    chunk_id is ``"{doc_id}-fig{i}"`` so figure chunks never collide with the
    document's text chunks (``"{doc_id}-{ordinal}"``).
    """
    chunks: list[Chunk] = []
    for i, img in enumerate(extract_figure_images(path)):
        text = describe_figure(img, client=client, model=model)
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{entry.id}-fig{i}",
                doc_id=entry.id,
                text=f"[Figure {i + 1}] {text}",
                source_title=entry.title,
                theme=entry.theme,
                section=f"Figure {i + 1}",
                url=entry.pdf_url,
            )
        )
    return chunks
