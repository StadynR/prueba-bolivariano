"""
Tests para el orquestador: clasificación de intent y routing.
Usa mocks para evitar llamadas reales a OpenAI y ChromaDB.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from app.orchestrator.graph import classify_intent, handle_unknown, consolidate
from app.agents.base_agent import AgentResponse


def _mock_llm_with_response(text: str):
    mock_response = MagicMock()
    mock_response.content = text
    return mock_response


# ─── classify_intent ──────────────────────────────────────────────────────────

@patch("app.orchestrator.graph.get_llm")
def test_classify_arquitectura(mock_get_llm):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = _mock_llm_with_response(
        json.dumps({"intents": ["arquitectura"]})
    )
    mock_get_llm.return_value = mock_instance

    state = {
        "query": "¿Qué debe tener un microservicio?",
        "intents": [], "agent_responses": [],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = classify_intent(state)
    assert "arquitectura" in result["intents"]


@patch("app.orchestrator.graph.get_llm")
def test_classify_unknown(mock_get_llm):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = _mock_llm_with_response(
        json.dumps({"intents": ["unknown"]})
    )
    mock_get_llm.return_value = mock_instance

    state = {
        "query": "¿Cuál es la capital de Francia?",
        "intents": [], "agent_responses": [],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = classify_intent(state)
    assert result["intents"] == ["unknown"]


@patch("app.orchestrator.graph.get_llm")
def test_classify_mixed(mock_get_llm):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = _mock_llm_with_response(
        json.dumps({"intents": ["arquitectura", "seguridad", "produccion"]})
    )
    mock_get_llm.return_value = mock_instance

    state = {
        "query": "Necesito publicar una nueva API que consume datos sensibles. ¿Qué debo cumplir?",
        "intents": [], "agent_responses": [],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = classify_intent(state)
    assert set(result["intents"]) == {"arquitectura", "seguridad", "produccion"}


@patch("app.orchestrator.graph.get_llm")
def test_classify_handles_invalid_json(mock_get_llm):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = _mock_llm_with_response("no es json valido {{{")
    mock_get_llm.return_value = mock_instance

    state = {
        "query": "pregunta rara",
        "intents": [], "agent_responses": [],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = classify_intent(state)
    # Fallback a unknown cuando JSON falla
    assert result["intents"] == ["unknown"]


# ─── handle_unknown ───────────────────────────────────────────────────────────

def test_handle_unknown_returns_standard_response():
    state = {
        "query": "¿Quién ganó el mundial?",
        "intents": ["unknown"], "agent_responses": [],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = handle_unknown(state)
    assert "no encontré información suficiente" in result["final_answer"].lower()
    assert result["agents_invoked"] == []
    assert len(result["warnings"]) > 0


# ─── consolidate ──────────────────────────────────────────────────────────────

@patch("app.orchestrator.graph.get_llm")
def test_consolidate_all_no_info(mock_get_llm):
    state = {
        "query": "¿Cuál es el origen del universo?",
        "intents": ["arquitectura"],
        "agent_responses": [
            AgentResponse(
                agent_name="Agente de Arquitectura",
                partial_answer="No encontré información suficiente en la base documental proporcionada.",
                sources=[],
                no_info_flag=True,
            )
        ],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = consolidate(state)
    assert "no encontré información suficiente" in result["final_answer"].lower()
    assert len(result["warnings"]) > 0


@patch("app.orchestrator.graph.get_llm")
def test_consolidate_deduplicates_sources(mock_get_llm):
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = _mock_llm_with_response("Respuesta consolidada.")
    mock_get_llm.return_value = mock_instance

    source = {"source_file": "01.txt", "section_title": "4. HEALTH CHECK", "doc_id": "DOC-01", "score": 0.9}
    state = {
        "query": "¿Qué microservicio necesita?",
        "intents": ["arquitectura"],
        "agent_responses": [
            AgentResponse(
                agent_name="Agente de Arquitectura",
                partial_answer="Debe tener health check.",
                sources=[source, source],  # duplicado intencional
                no_info_flag=False,
            )
        ],
        "final_answer": "", "agents_invoked": [], "sources": [], "warnings": [],
    }
    result = consolidate(state)
    assert len(result["sources"]) == 1
