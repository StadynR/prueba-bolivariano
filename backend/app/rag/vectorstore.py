"""
Inicialización y acceso al cliente ChromaDB.

Diseño de colecciones:
  - Una colección por documento fuente → base para control de acceso documental.
  - Metadatos por chunk: source_file, doc_id, section_title, chunk_index.

Esto permite, en el futuro, filtrar por perfil de usuario sin cambiar la
arquitectura (solo agregar filtro de metadatos en la búsqueda).
"""
from __future__ import annotations
import os
# Desactivar telemetría de ChromaDB antes de que el módulo inicialice PostHog.
# ChromaSettings(anonymized_telemetry=False) no es suficiente en 0.5.x porque
# el cliente PostHog ya está instanciado en tiempo de importación.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

# Nombres de colecciones — clave para mapear agente ↔ colección
COLLECTION_ARCHITECTURE = "arquitectura"
COLLECTION_SECURITY = "seguridad"
COLLECTION_PRODUCTION = "produccion"

ALL_COLLECTIONS = [
    COLLECTION_ARCHITECTURE,
    COLLECTION_SECURITY,
    COLLECTION_PRODUCTION,
]

_client: chromadb.PersistentClient | None = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """Devuelve una colección existente (debe haber sido creada por ingest.py)."""
    client = get_chroma_client()
    return client.get_collection(name=name)


def get_or_create_collection(name: str) -> chromadb.Collection:
    """Crea la colección si no existe. Usado por ingest.py."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
