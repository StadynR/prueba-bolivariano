"""
Configuración centralizada mediante Pydantic Settings.
Las API keys se cargan ÚNICAMENTE desde variables de entorno o el archivo .env
en la raíz del repositorio — nunca se registran en logs.
"""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

# Raíz del repositorio: config.py está en backend/app/, subimos dos niveles
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenAI ───────────────────────────────────────────────────
    openai_api_key: str = Field(...)
    llm_model: str = Field("gpt-4o-mini")
    embedding_model: str = Field("text-embedding-3-small")

    # ── ChromaDB ─────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_data")

    # ── Retrieval ─────────────────────────────────────────────────
    rag_top_k: int = Field(4)
    rag_relevance_threshold: float = Field(0.45)

    # ── Servidor ─────────────────────────────────────────────────
    log_level: str = Field("INFO")
    cors_origins: str = Field("http://localhost:8000,http://127.0.0.1:8000")

    @model_validator(mode="after")
    def resolve_chroma_path(self) -> "Settings":
        # Resolver chroma_persist_dir como ruta absoluta desde _REPO_ROOT
        # evita que una ruta relativa apunte a distintos lugares según el CWD
        chroma_path = Path(self.chroma_persist_dir)
        if not chroma_path.is_absolute():
            self.chroma_persist_dir = str(_REPO_ROOT / chroma_path)
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
