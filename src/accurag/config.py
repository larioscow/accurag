from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # repo root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ACCURAG_",
        env_file=".env",
        extra="ignore",
        env_ignore_empty=True,
        # Lets fields be set by their Python name (in addition to their env
        # alias, which is read by default), so a caller can do
        # Settings(openai_api_key=...) instead of only the OPENAI_API_KEY env var.
        validate_by_name=True,
    )
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    cohere_api_key: str = Field(default="", validation_alias="COHERE_API_KEY")
    embed_model: str = "text-embedding-3-large"
    embed_dim: int = 3072
    local_embed_model: str = "BAAI/bge-small-en-v1.5"  # for LocalEmbeddingClient (no-API)
    vision_model: str = (
        "claude-sonnet-4-6"  # figure extraction; reads complex charts (gpt-4o-mini misreads them)
    )
    collection: str = "accurag_dense"

    manifest_path: Path = ROOT / "corpus" / "manifest.json"
    raw_dir: Path = ROOT / "data" / "raw"
    parse_cache_path: Path = (
        ROOT / "data" / "parse_cache.jsonl"
    )  # checkpoint: parsed chunks (no vectors)
    chunks_path: Path = ROOT / "data" / "chunks.jsonl"
    qdrant_path: Path = ROOT / "data" / "qdrant"

    chunk_size: int = 512  # target chunk size in tokens (clamped to the embedder's window)
    rerank_candidates: int = 20  # candidates over-fetched before reranking; deeper (50) measured
    #                              no better on this corpus (see docs/EVAL_RESULTS.md). Tunable.


settings = Settings()
