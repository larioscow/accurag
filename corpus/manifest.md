# accurag Corpus Manifest

**Snapshot:** 2026-05-30 · **47 documents** · frozen & reproducible.

A hand-curated AI-automation corpus: recent agent / retrieval / RAG / evaluation papers + open vendor reports. Every arXiv ID was web-verified against its live abstract page before commit; the two post-cutoff 2026 papers were re-verified manually. We commit this manifest + a fetch script (`corpus/fetch.py`, planned) — **never the PDFs themselves** (legal, small, reproducible).

- **~38 arXiv PDFs** + **9 vendor reports** (1 PDF, 8 HTML pages → mixed-format ingestion).
- Snapshot is immutable so the eval golden-set stays reproducible.

## RAG foundations & long-context (7)

| # | Title | Authors | Year | ID |
|---|---|---|---|---|
| 1 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 | [2005.11401](https://arxiv.org/abs/2005.11401) |
| 2 | RAG for Large Language Models: A Survey | Gao et al. | 2024 | [2312.10997](https://arxiv.org/abs/2312.10997) |
| 3 | Lost in the Middle: How LMs Use Long Contexts | Liu et al. | 2023 | [2307.03172](https://arxiv.org/abs/2307.03172) |
| 4 | REALM: Retrieval-Augmented LM Pre-Training | Guu et al. | 2020 | [2002.08909](https://arxiv.org/abs/2002.08909) |
| 5 | Improving LMs by Retrieving from Trillions of Tokens (RETRO) | Borgeaud et al. | 2022 | [2112.04426](https://arxiv.org/abs/2112.04426) |
| 6 | RAG or Long-Context LLMs? A Comprehensive Study + Hybrid | Li et al. | 2024 | [2407.16833](https://arxiv.org/abs/2407.16833) |
| 7 | In Defense of RAG in the Era of Long-Context LLMs | Yu et al. | 2024 | [2409.01666](https://arxiv.org/abs/2409.01666) |

## Retrieval / hybrid / rerank (9)

| # | Title | Authors | Year | ID |
|---|---|---|---|---|
| 8 | Dense Passage Retrieval (DPR) | Karpukhin et al. | 2020 | [2004.04906](https://arxiv.org/abs/2004.04906) |
| 9 | ColBERT: Late Interaction over BERT | Khattab & Zaharia | 2020 | [2004.12832](https://arxiv.org/abs/2004.12832) |
| 10 | ColBERTv2: Lightweight Late Interaction | Santhanam et al. | 2022 | [2112.01488](https://arxiv.org/abs/2112.01488) |
| 11 | Precise Zero-Shot Dense Retrieval (HyDE) | Gao et al. | 2022 | [2212.10496](https://arxiv.org/abs/2212.10496) |
| 12 | Reciprocal Rank Fusion (RRF) | Cormack et al. | 2009 | doi:[10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114) |
| 13 | SPLADE: Sparse Lexical & Expansion | Formal et al. | 2021 | [2107.05720](https://arxiv.org/abs/2107.05720) |
| 14 | C-Pack / BGE Embeddings | Xiao et al. | 2023 | [2309.07597](https://arxiv.org/abs/2309.07597) |
| 15 | Document Ranking with seq2seq (monoT5) | Nogueira et al. | 2020 | [2003.06713](https://arxiv.org/abs/2003.06713) |
| 16 | From BM25 to Corrective RAG: Text-and-Table Benchmarking ⭐ | Akarsu et al. | 2026 | [2604.01733](https://arxiv.org/abs/2604.01733) |

## Agentic & adaptive patterns (9)

| # | Title | Authors | Year | ID |
|---|---|---|---|---|
| 17 | ReAct: Reasoning + Acting | Yao et al. | 2022 | [2210.03629](https://arxiv.org/abs/2210.03629) |
| 18 | Toolformer | Schick et al. | 2023 | [2302.04761](https://arxiv.org/abs/2302.04761) |
| 19 | Reflexion | Shinn et al. | 2023 | [2303.11366](https://arxiv.org/abs/2303.11366) |
| 20 | Self-RAG | Asai et al. | 2023 | [2310.11511](https://arxiv.org/abs/2310.11511) |
| 21 | Corrective RAG (CRAG) | Yan et al. | 2024 | [2401.15884](https://arxiv.org/abs/2401.15884) |
| 22 | Adaptive-RAG (complexity routing) | Jeong et al. | 2024 | [2403.14403](https://arxiv.org/abs/2403.14403) |
| 23 | Active RAG (FLARE) | Jiang et al. | 2023 | [2305.06983](https://arxiv.org/abs/2305.06983) |
| 24 | ReWOO: Decoupling Reasoning from Observations | Xu et al. | 2023 | [2305.18323](https://arxiv.org/abs/2305.18323) |
| 25 | Agentic RAG: A Survey | Singh et al. | 2025 | [2501.09136](https://arxiv.org/abs/2501.09136) |

## Chunking & parsing (7)

| # | Title | Authors | Year | ID |
|---|---|---|---|---|
| 26 | Evaluating Chunking Strategies for Retrieval | Smith & Troynikov (Chroma) | 2024 | [web](https://research.trychroma.com/evaluating-chunking) |
| 27 | Late Chunking | Günther et al. (Jina) | 2024 | [2409.04701](https://arxiv.org/abs/2409.04701) |
| 28 | Docling Technical Report | Auer et al. (IBM) | 2024 | [2408.09869](https://arxiv.org/abs/2408.09869) |
| 29 | Dense X Retrieval (propositions) | Chen et al. | 2023 | [2312.06648](https://arxiv.org/abs/2312.06648) |
| 30 | Nougat: Neural Doc Understanding | Blecher et al. | 2023 | [2308.13418](https://arxiv.org/abs/2308.13418) |
| 31 | Systematic Analysis of Chunking Strategies ⭐ | Bennani & Moslonka | 2026 | [2601.14123](https://arxiv.org/abs/2601.14123) |
| 32 | DocLayNet: Layout-Analysis Dataset | Pfitzmann et al. (IBM) | 2022 | [2206.01062](https://arxiv.org/abs/2206.01062) |

## Evaluation & faithfulness (8)

| # | Title | Authors | Year | ID |
|---|---|---|---|---|
| 33 | Ragas | Es et al. | 2023 | [2309.15217](https://arxiv.org/abs/2309.15217) |
| 34 | ARES | Saad-Falcon et al. | 2023 | [2311.09476](https://arxiv.org/abs/2311.09476) |
| 35 | RAGChecker | Ru et al. | 2024 | [2408.08067](https://arxiv.org/abs/2408.08067) |
| 36 | Can we Evaluate RAGs with Synthetic Data? | van Elburg et al. | 2025 | [2508.11758](https://arxiv.org/abs/2508.11758) |
| 37 | Benchmarking LLMs in RAG (RGB) | Chen et al. | 2023 | [2309.01431](https://arxiv.org/abs/2309.01431) |
| 38 | CRUD-RAG | Lyu et al. | 2024 | [2401.17043](https://arxiv.org/abs/2401.17043) |
| 39 | FaithBench (hallucination) | Bao et al. (Vectara) | 2024 | [2410.13210](https://arxiv.org/abs/2410.13210) |
| 40 | Correctness is not Faithfulness in RAG Attributions | Wallat et al. | 2024 | [2412.18004](https://arxiv.org/abs/2412.18004) |

## Vendor reports + deliberately-skipped (7)

| # | Title | Authors | Year | Source |
|---|---|---|---|---|
| 41 | Introducing Contextual Retrieval (+ appendix PDF) | Anthropic | 2024 | [pdf](https://assets.anthropic.com/m/1632cded0a125333/original/Contextual-Retrieval-Appendix-2.pdf) |
| 42 | Building Effective Agents | Anthropic | 2024 | [web](https://www.anthropic.com/research/building-effective-agents) |
| 43 | GraphRAG: From Local to Global — *evaluated & skipped* | Edge et al. (Microsoft) | 2024 | [2404.16130](https://arxiv.org/abs/2404.16130) |
| 44 | Introducing Rerank 3 | Cohere | 2024 | [web](https://cohere.com/blog/rerank-3) |
| 45 | New Embedding Models (text-embedding-3) | OpenAI | 2024 | [web](https://openai.com/index/new-embedding-models-and-api-updates/) |
| 46 | BM42: New Baseline for Hybrid Search | Vasnetsov (Qdrant) | 2024 | [web](https://qdrant.tech/articles/bm42/) |
| 47 | voyage-3 Embedding Models | Voyage AI | 2024 | [web](https://blog.voyageai.com/2024/09/18/voyage-3/) |

⭐ = post-training-cutoff 2026 paper, manually verified against the live arXiv page.

> Machine-readable source of truth: [`corpus/manifest.json`](manifest.json).
