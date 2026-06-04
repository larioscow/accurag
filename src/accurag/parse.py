"""Document parsing via Docling.

A single module-level DocumentConverter is created lazily on first use so that
importing this module never triggers a heavy Docling import at package load time
and so that tests that only probe the function signature can import without
Docling installed.

Usage::

    from pathlib import Path
    from accurag.parse import parse_file

    doc = parse_file(Path("paper.pdf"))
    print(doc.export_to_markdown())
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type hints; never executed at runtime during module load.
    from docling.datamodel.document import DoclingDocument

# Module-level converter — created once, reused across all calls.
_converter = None


def _get_converter():
    """Return the shared DocumentConverter, instantiating it on first call.

    Docling is pinned to the **CPU** accelerator. This project is CPU-only by
    design, and it is also a correctness requirement on Apple Silicon: the layout
    model (rt_detr_v2) builds float64 position embeddings, and the MPS backend
    raises ``Cannot convert a MPS Tensor to float64`` — so auto-selecting MPS
    fails every PDF. CPU avoids that entirely.
    """
    global _converter  # noqa: PLW0603
    if _converter is None:
        # All lazy — keeps module import cheap and Docling-free for unit tests.
        try:
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
                TableFormerMode,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Parser deps not installed — run: pip install 'accurag[ingest]'"
            ) from exc

        pipeline_options = PdfPipelineOptions()
        # CPU + bounded threads. The "no GPU ever" rule plus the MPS-float64
        # crash on Apple Silicon (see docstring) make CPU mandatory; 4 threads
        # keeps memory predictable while staying responsive.
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=4
        )
        # The corpus is digital-born (arxiv PDFs + vendor HTML) with real text
        # layers, so OCR adds large CPU cost for no gain. Table structure stays
        # on — tables are where naive RAG loses information — but in FAST mode,
        # which is markedly cheaper than the default ACCURATE TableFormer.
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        # Don't retain rendered bitmaps. Page/picture images balloon peak memory
        # and we never embed images — text + table structure is all we chunk.
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        _converter = DocumentConverter(
            format_options={
                # The pypdfium2 backend is the big memory lever: ~2.5GB vs ~6GB
                # peak and roughly 2x faster than the default native backend.
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )
    return _converter


def parse_file(path: Path) -> DoclingDocument:
    """Convert *path* to a :class:`~docling.datamodel.document.DoclingDocument`.

    The module-level ``DocumentConverter`` is reused across calls to avoid
    repeated initialisation overhead (model loading etc.).

    Args:
        path: Absolute or relative path to the document to parse (PDF, DOCX,
              HTML, XML, CSV — anything Docling supports).

    Returns:
        A :class:`DoclingDocument` ready for chunking or export.
    """
    converter = _get_converter()
    return converter.convert(str(path)).document
