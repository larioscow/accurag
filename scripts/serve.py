"""Thin HTTP wrapper around accurag's RagPipeline.ask().

Exposes the RAG as a small JSON API so external clients (e.g. an n8n workflow
running in Docker) can query it without importing the heavy retrieval stack.

Run:
    uv run uvicorn scripts.serve:app --host 0.0.0.0 --port 8077
    # or: .venv/bin/python -m uvicorn scripts.serve:app --host 0.0.0.0 --port 8077

Endpoints:
    GET  /health        -> {"status": "ok"}
    POST /ask           -> {"answer": str, "sources": [...], "strategy", "k"}
                           body: {"query": str, "strategy"?: str, "k"?: int}

The pipeline is built once at startup and reused across requests.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from accurag import RagPipeline

Strategy = Literal["dense", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: Strategy = "hybrid_rerank"
    k: int = Field(5, ge=1, le=20)


class SourceCard(BaseModel):
    rank: int
    score: float
    title: str
    url: str | None = None
    section: str | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCard]
    strategy: str
    k: int


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    # Built once, reused. Uses .env keys (OpenAI/Anthropic/Cohere).
    return RagPipeline()


app = FastAPI(title="accurag API", version="1.0.0")

# Allow the demo website (any localhost origin) to call this directly if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    rag = get_pipeline()
    answer = rag.ask(req.query, strategy=req.strategy, k=req.k)
    sources = [
        SourceCard(
            rank=s.rank,
            score=round(s.score, 4),
            title=s.chunk.source_title,
            url=getattr(s.chunk, "url", None),
            section=getattr(s.chunk, "section", None),
            snippet=s.chunk.text[:300],
        )
        for s in answer.sources
    ]
    return AskResponse(
        answer=answer.text, sources=sources, strategy=req.strategy, k=req.k
    )
