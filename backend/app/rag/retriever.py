"""
Capa de recuperación semántica sobre ChromaDB.

Cada llamada a `retrieve` devuelve chunks con:
  - texto del chunk
  - metadatos (source_file, section_title, doc_id)
  - distancia coseno (0 = idéntico, 2 = opuesto; usamos 1-distance como score)

El umbral de relevancia se configura en RAG_RELEVANCE_THRESHOLD (.env).
Si ningún chunk supera el umbral, el agente debe activar no_info_flag.
"""
from __future__ import annotations
from dataclasses import dataclass
from app.rag.vectorstore import get_collection
from app.config import settings
from app.llm_factory import get_embeddings


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    section_title: str
    doc_id: str
    score: float  # 0-1, mayor = más relevante


def retrieve(
    query: str,
    collection_name: str,
    k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Recupera los k chunks más relevantes de una colección dada.

    Args:
        query: Pregunta del usuario.
        collection_name: Nombre de la colección ChromaDB a consultar.
        k: Número de chunks. Por defecto usa settings.rag_top_k.

    Returns:
        Lista de RetrievedChunk ordenada por relevancia descendente,
        filtrada por settings.rag_relevance_threshold.
    """
    k = k or settings.rag_top_k
    collection = get_collection(collection_name)
    embeddings = get_embeddings()

    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine distance: 0 = idéntico → score = 1 - distance
        score = max(0.0, 1.0 - dist)
        if score >= settings.rag_relevance_threshold:
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source_file=meta.get("source_file", ""),
                    section_title=meta.get("section_title", ""),
                    doc_id=meta.get("doc_id", ""),
                    score=round(score, 4),
                )
            )

    return chunks
