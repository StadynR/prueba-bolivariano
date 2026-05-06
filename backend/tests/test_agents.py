"""
Tests unitarios para los agentes especializados.
Usa mocks para OpenAI y ChromaDB — no requiere credenciales reales.
"""
import pytest
from unittest.mock import MagicMock, patch
from app.agents.architecture_agent import ArchitectureAgent
from app.agents.security_agent import SecurityAgent
from app.agents.production_agent import ProductionAgent
from app.agents.base_agent import AgentResponse
from app.rag.retriever import RetrievedChunk


MOCK_CHUNKS = [
    RetrievedChunk(
        text="Todo microservicio debe tener health check implementado.",
        source_file="01_estandares_arquitectura_microservicios.txt",
        section_title="4. HEALTH CHECK Y DISPONIBILIDAD",
        doc_id="DOC-01",
        score=0.85,
    )
]


def _make_llm_response(text: str):
    """Crea un mock de respuesta de ChatOpenAI."""
    mock_response = MagicMock()
    mock_response.content = text
    mock_response.usage_metadata = {"total_tokens": 50}
    return mock_response


@patch("app.agents.base_agent.retrieve")
@patch("app.agents.base_agent.get_llm")
def test_architecture_agent_returns_answer(mock_get_llm, mock_retrieve):
    mock_retrieve.return_value = MOCK_CHUNKS
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = _make_llm_response(
        "El microservicio debe implementar un endpoint de health check."
    )
    mock_get_llm.return_value = mock_llm_instance

    agent = ArchitectureAgent()
    result: AgentResponse = agent.run("¿Qué debe tener un microservicio?")

    assert result.agent_name == "Agente de Arquitectura"
    assert result.no_info_flag is False
    assert len(result.sources) == 1
    assert result.sources[0]["doc_id"] == "DOC-01"


@patch("app.agents.base_agent.retrieve")
def test_agent_no_info_when_no_chunks(mock_retrieve):
    mock_retrieve.return_value = []

    agent = SecurityAgent()
    result: AgentResponse = agent.run("¿Cuál es el mejor restaurante?")

    assert result.no_info_flag is True
    assert "no encontré información suficiente" in result.partial_answer.lower()
    assert result.sources == []


@patch("app.agents.base_agent.retrieve")
@patch("app.agents.base_agent.get_llm")
def test_agent_no_info_flag_when_llm_says_no_info(mock_get_llm, mock_retrieve):
    mock_retrieve.return_value = MOCK_CHUNKS
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = _make_llm_response(
        "No encontré información suficiente en la base documental proporcionada."
    )
    mock_get_llm.return_value = mock_llm_instance

    agent = ProductionAgent()
    result: AgentResponse = agent.run("¿Qué es el universo?")

    assert result.no_info_flag is True


@patch("app.agents.base_agent.retrieve")
@patch("app.agents.base_agent.get_llm")
def test_security_agent_sources_populated(mock_get_llm, mock_retrieve):
    mock_retrieve.return_value = MOCK_CHUNKS
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = _make_llm_response(
        "No se debe registrar información sensible en logs."
    )
    mock_get_llm.return_value = mock_llm_instance

    agent = SecurityAgent()
    result = agent.run("¿Puedo loguear el número de cédula?")

    assert not result.no_info_flag
    assert any(s["source_file"] == "01_estandares_arquitectura_microservicios.txt" for s in result.sources)
