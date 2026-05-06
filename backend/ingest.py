"""
Script de ingesta: lee los 3 documentos TXT, los divide en chunks por sección
y los inserta en sus respectivas colecciones ChromaDB.

Ejecutar UNA VEZ antes de iniciar el servidor:
    cd backend/
    python ingest.py

Chunking strategy:
  - Se detectan secciones por líneas que comienzan con dígito + punto (ej: "3. LINEAMIENTOS").
  - Cada sección se almacena como un chunk independiente con sus metadatos.
  - Esto preserva la semántica del documento y facilita citar la sección exacta.
"""
import re
import sys
import os
import logging
from pathlib import Path

# Directorio donde vive este script (backend/)
_BACKEND_DIR = Path(__file__).resolve().parent

# Asegurar que el path de app esté disponible
sys.path.insert(0, str(_BACKEND_DIR))

from app.rag.vectorstore import get_or_create_collection
from app.config import settings
from app.llm_factory import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Mapeo: colección → archivo fuente (rutas relativas a backend/)
DOCS_CONFIG = {
    "arquitectura": {
        "file": str(_BACKEND_DIR / "docs" / "01_estandares_arquitectura_microservicios.txt"),
        "doc_id": "DOC-01",
    },
    "seguridad": {
        "file": str(_BACKEND_DIR / "docs" / "02_lineamientos_seguridad_apis.txt"),
        "doc_id": "DOC-02",
    },
    "produccion": {
        "file": str(_BACKEND_DIR / "docs" / "03_checklist_paso_produccion.txt"),
        "doc_id": "DOC-03",
    },
}

# Patrón que identifica inicio de una sección numerada (ej: "3. LINEAMIENTOS")
SECTION_PATTERN = re.compile(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]")


def split_into_sections(text: str) -> list[dict]:
    """
    Divide el texto en secciones basándose en headers numerados.
    Retorna lista de dicts con 'title' y 'content'.
    """
    lines = text.splitlines()
    sections = []
    current_title = "ENCABEZADO"
    current_lines: list[str] = []

    for line in lines:
        if SECTION_PATTERN.match(line.strip()):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append({"title": current_title, "content": content})
            current_title = line.strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Último bloque
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({"title": current_title, "content": content})

    return sections


def ingest_document(collection_name: str, file_path: str, doc_id: str) -> int:
    """Procesa un documento e inserta sus chunks en ChromaDB. Retorna nº chunks."""
    if not os.path.exists(file_path):
        logger.error("Archivo no encontrado: %s", file_path)
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = split_into_sections(text)
    logger.info(
        "[%s] %d secciones detectadas en %s",
        collection_name,
        len(sections),
        file_path,
    )

    embeddings_fn = get_embeddings()
    collection = get_or_create_collection(collection_name)

    documents = [s["content"] for s in sections]
    metadatas = [
        {
            "source_file": os.path.basename(file_path),
            "section_title": s["title"],
            "doc_id": doc_id,
            "chunk_index": idx,
        }
        for idx, s in enumerate(sections)
    ]
    ids = [f"{doc_id}-chunk-{idx}" for idx in range(len(sections))]

    # Generar embeddings en batch
    logger.info("[%s] Generando embeddings...", collection_name)
    embeddings = embeddings_fn.embed_documents(documents)

    # Upsert para que sea idempotente (re-ejecutar no duplica)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("[%s] %d chunks insertados correctamente.", collection_name, len(sections))
    return len(sections)


def main() -> None:
    logger.info("=== Iniciando ingesta de documentos ===")
    logger.info("ChromaDB persist dir: %s", settings.chroma_persist_dir)
    total = 0
    for collection_name, cfg in DOCS_CONFIG.items():
        count = ingest_document(collection_name, cfg["file"], cfg["doc_id"])
        total += count
    logger.info("=== Ingesta completada: %d chunks totales ===", total)


if __name__ == "__main__":
    main()
