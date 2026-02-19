"""
Central configuration management for EcoloGRAPH.

Loads settings from environment variables and provides typed access.
"""
import os
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """LLM configuration with dual-model support.
    
    Two models serve different purposes:
      - ingestion_model: VLM for building the knowledge graph (handles PDFs, images, non-OCR docs)
      - reasoning_model: Powerful text model for traversing the graph and answering queries
    
    The 'model' field is kept for backward compatibility and defaults to the ingestion model.
    """
    provider: str = Field(default="local", alias="LLM_PROVIDER")
    base_url: str = Field(default="http://localhost:11434/v1", alias="LOCAL_LLM_BASE_URL")
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    
    # Dual-model configuration
    ingestion_model: str = Field(
        default="qwen3:8b",
        alias="INGESTION_LLM_MODEL",
        description="Model for entity extraction and knowledge graph construction"
    )
    reasoning_model: str = Field(
        default="gpt-oss:20b",
        alias="REASONING_LLM_MODEL",
        description="Text model for graph traversal, agent queries, and chat"
    )
    
    # Backward compatibility: 'model' defaults to ingestion_model
    model: str = Field(default="gpt-oss:20b", alias="LOCAL_LLM_MODEL")


class EmbeddingSettings(BaseSettings):
    """Embedding configuration."""
    provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")


class QdrantSettings(BaseSettings):
    """Qdrant vector store configuration."""
    host: str = Field(default="localhost", alias="QDRANT_HOST")
    port: int = Field(default=6333, alias="QDRANT_PORT")
    collection: str = Field(default="ecolograh_chunks", alias="QDRANT_COLLECTION")


class Neo4jSettings(BaseSettings):
    """Neo4j graph database configuration."""
    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: str = Field(default="password", alias="NEO4J_PASSWORD")


class RateLimitSettings(BaseSettings):
    """Rate limits for external APIs (seconds between requests)."""
    crossref: float = Field(default=1.0, alias="CROSSREF_RATE_LIMIT")
    semantic_scholar: float = Field(default=1.0, alias="SEMANTIC_SCHOLAR_RATE_LIMIT")
    fishbase: float = Field(default=2.0, alias="FISHBASE_RATE_LIMIT")
    gbif: float = Field(default=1.0, alias="GBIF_RATE_LIMIT")
    worms: float = Field(default=1.0, alias="WORMS_RATE_LIMIT")


class PathSettings(BaseSettings):
    """Path configuration."""
    data_raw: Path = Field(default=Path("data/raw"), alias="DATA_RAW_PATH")
    data_processed: Path = Field(default=Path("data/processed"), alias="DATA_PROCESSED_PATH")
    data_cache: Path = Field(default=Path("data/cache"), alias="DATA_CACHE_PATH")
    
    def resolve(self, base_dir: Path) -> "PathSettings":
        """Resolve relative paths against base directory."""
        return PathSettings(
            data_raw=base_dir / self.data_raw,
            data_processed=base_dir / self.data_processed,
            data_cache=base_dir / self.data_cache
        )


class Settings(BaseSettings):
    """Main settings aggregator."""
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # Project root
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def _load_api_key_file():
    """Load API key from config/api-key file if it exists."""
    key_path = Path(__file__).parent.parent.parent / "config" / "api-key"
    if key_path.exists() and not os.environ.get("OPENAI_API_KEY"):
        key = key_path.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    _load_api_key_file()
    return Settings()


# Convenience function
def load_dotenv_if_exists():
    """Load .env file if it exists, then load config/api-key."""
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _load_api_key_file()
