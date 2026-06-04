import json

from accurag.config import settings
from accurag.models import Chunk, ManifestEntry


def test_manifest_entry_parses_real_manifest():
    data = json.loads(settings.manifest_path.read_text())
    entries = [ManifestEntry(**d) for d in data["documents"]]
    assert len(entries) == 47
    assert all(e.pdf_url.startswith("http") for e in entries)


def test_chunk_id_is_doc_id_plus_ordinal():
    c = Chunk(
        chunk_id="3-0",
        doc_id=3,
        text="hello",
        source_title="Lost in the Middle",
        theme="rag_foundations",
        section=None,
        url="https://arxiv.org/pdf/2307.03172",
    )
    assert c.chunk_id == "3-0"
    assert c.doc_id == 3
