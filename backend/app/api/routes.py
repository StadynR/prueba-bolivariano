"""
Rutas de la API REST.

Endpoints:
  POST /api/query  — Recibe pregunta y devuelve respuesta consolidada.
  GET  /api/health — Health check del servicio.

Logging seguro:
  - Se registra query_id, timestamp, agentes invocados y tokens estimados.
  - NO se registra el contenido completo de la pregunta ni de la respuesta
    (podría contener información sensible del usuario o del banco).
"""
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse, SourceReference
from app.orchestrator.graph import orchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Recibe la pregunta del usuario y la procesa a través del orquestador LangGraph.
    Devuelve respuesta consolidada con trazabilidad de agentes y fuentes.
    """
    query_id = str(uuid.uuid4())

    # Log seguro: solo metadatos, nunca el contenido de la pregunta
    logger.info(
        "query_received | query_id=%s | question_len=%d | timestamp=%s",
        query_id,
        len(request.question),
        datetime.now(timezone.utc).isoformat(),
    )

    try:
        initial_state = {
            "query": request.question,
            "intents": [],
            "agent_responses": [],
            "final_answer": "",
            "agents_invoked": [],
            "sources": [],
            "warnings": [],
        }

        result = orchestrator.invoke(initial_state)

        sources = [
            SourceReference(
                source_file=s["source_file"],
                section_title=s["section_title"],
                doc_id=s["doc_id"],
                score=s["score"],
            )
            for s in result.get("sources", [])
        ]

        logger.info(
            "query_completed | query_id=%s | agents=%s | sources=%d | warnings=%d",
            query_id,
            result.get("agents_invoked", []),
            len(sources),
            len(result.get("warnings", [])),
        )

        return QueryResponse(
            answer=result["final_answer"],
            agents_invoked=result.get("agents_invoked", []),
            sources=sources,
            warnings=result.get("warnings", []),
            query_id=query_id,
        )

    except Exception as exc:
        logger.error(
            "query_error | query_id=%s | error_type=%s | error=%s",
            query_id,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.") from exc
