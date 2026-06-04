"""Chunk-size sweep on the pymupdf (token-window) path: 256 / 512 / 800.

Ingests the corpus at each size into its own collection, then scores retrieval
(dense + hybrid) on the doc-level golden set. Shared qdrant/embed/sparse clients
avoid the local-Qdrant single-access lock and reloading SPLADE three times.
"""
import json
from pathlib import Path
from accurag import RagPipeline, GoldenQA
from accurag import index, embed, sparse_embed
from accurag.config import settings, ROOT

golden = [GoldenQA(**json.loads(l)) for l in Path("data/golden/golden.jsonl").read_text().splitlines() if l.strip()]
qclient = index.make_client()
eclient = embed.make_client()
smodel = sparse_embed.make_sparse_model()

sizes = [256, 512, 800]
summary = []
for size in sizes:
    settings.collection = f"accurag_pm_{size}"
    settings.parse_cache_path = ROOT / "data" / f"cache_pm_{size}.jsonl"
    settings.chunks_path = ROOT / "data" / f"chunks_pm_{size}.jsonl"
    settings.parse_cache_path.unlink(missing_ok=True)
    settings.chunks_path.unlink(missing_ok=True)

    rag = RagPipeline(qdrant_client=qclient, embed_client=eclient, sparse_model=smodel)
    n = rag.ingest(parser="pymupdf", chunk_size=size)
    rep = rag.evaluate(golden, strategies=("dense", "hybrid"), k=5)
    print(f"\n##### chunk_size={size}  ({n} chunks)", flush=True)
    print(rep.to_markdown(), flush=True)
    d = {r.strategy: r for r in rep.rows}
    summary.append((size, n, d["dense"].recall_at_k, d["dense"].mrr, d["hybrid"].recall_at_k, d["hybrid"].mrr))

print("\n\n===== SIZE SWEEP SUMMARY (pymupdf, doc-level golden) =====", flush=True)
print("| chunk_size | #chunks | dense Recall@5 | dense MRR | hybrid Recall@5 | hybrid MRR |", flush=True)
print("|---|---|---|---|---|---|", flush=True)
for size, n, dr, dm, hr, hm in summary:
    print(f"| {size} | {n} | {dr:.3f} | {dm:.3f} | {hr:.3f} | {hm:.3f} |", flush=True)
print("\n[sweep] DONE", flush=True)
