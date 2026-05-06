"""
Orquestador principal implementado con LangGraph StateGraph.

Flujo del grafo:
  classify_intent → (condicional)
      ├── run_agents     → consolidate → END
      └── handle_unknown              → END

Nodos:
  - classify_intent : LLM determina qué dominios son relevantes en la pregunta.
  - run_agents      : Invoca en paralelo solo los agentes necesarios (Send API).
  - consolidate     : LLM genera respuesta final unificada con fuentes.
  - handle_unknown  : Respuesta estándar para preguntas fuera del alcance.

Decisión de routing:
  - Se usa un LLM de clasificación con output estructurado para decidir qué dominios
    aplican: ["arquitectura", "seguridad", "produccion"] o ["unknown"].
  - Esto permite consultas mixtas (varios agentes) sin lógica de keywords frágil.
  - El resultado es trazable y auditable en el estado del grafo.
"""
from __future__ import annotations

import logging
import json
from typing import TypedDict, Annotated
import operator

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.agents.base_agent import AgentResponse
from app.agents.architecture_agent import ArchitectureAgent
from app.agents.security_agent import SecurityAgent
from app.agents.production_agent import ProductionAgent
from app.config import settings
from app.llm_factory import get_llm

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """
    Parsea JSON de una respuesta LLM que puede venir envuelta en un bloque
    markdown (```json ... ```) o como JSON puro.
    """
    stripped = text.strip()
    # Quitar bloque de código markdown si existe
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Eliminar primera línea (```json o ```) y última (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(inner).strip()
    return json.loads(stripped)


# ─── Agentes disponibles ──────────────────────────────────────────────────────
AGENT_REGISTRY: dict[str, type] = {
    "arquitectura": ArchitectureAgent,
    "seguridad": SecurityAgent,
    "produccion": ProductionAgent,
}

VALID_INTENTS = list(AGENT_REGISTRY.keys())

# ─── Estado del grafo ─────────────────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    intents: list[str]                        # ["arquitectura", "seguridad", ...]
    agent_responses: Annotated[list[AgentResponse], operator.add]  # fan-in
    final_answer: str
    agents_invoked: list[str]
    sources: list[dict]
    warnings: list[str]


# ─── Nodos ────────────────────────────────────────────────────────────────────

def classify_intent(state: GraphState) -> GraphState:
    """
    Usa LLM para identificar qué dominios cubre la pregunta.
    Retorna lista de intents válidos o ["unknown"].
    """
    query = state["query"]

    llm = get_llm()

    system = (
        "Eres un clasificador de intención para una mesa de ayuda técnica bancaria. "
        "Los dominios disponibles son:\n"
        "  - arquitectura: estándares de microservicios, diseño de APIs, health checks, "
        "trazabilidad, manejo de errores, observabilidad, pruebas técnicas.\n"
        "  - seguridad: controles de seguridad, autenticación, autorización, manejo de secretos, "
        "protección de datos sensibles, auditoría, guardrails en IA.\n"
        "  - produccion: criterios de paso a producción, checklists, pruebas requeridas, "
        "aprobaciones, plan de despliegue, plan de rollback, monitoreo post-despliegue.\n\n"
        "Responde ÚNICAMENTE con un JSON válido con la clave 'intents' que contenga "
        "una lista de los dominios relevantes. "
        "Si la consulta no corresponde a ningún dominio, usa [\"unknown\"]. "
        "Ejemplos:\n"
        '  {"intents": ["arquitectura"]}\n'
        '  {"intents": ["seguridad", "produccion"]}\n'
        '  {"intents": ["arquitectura", "seguridad", "produccion"]}\n'
        '  {"intents": ["unknown"]}'
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Clasifica la siguiente consulta: {query}"),
    ])

    logger.debug("classify_intent: respuesta cruda del LLM: %r", response.content)

    try:
        data = _parse_json_response(response.content)
        intents = data.get("intents", ["unknown"])
        # Validar que solo contenga valores permitidos
        intents = [i for i in intents if i in VALID_INTENTS + ["unknown"]]
        if not intents:
            intents = ["unknown"]
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "classify_intent: fallo al parsear JSON (%s) — respuesta: %r — usando unknown",
            exc,
            response.content,
        )
        intents = ["unknown"]

    logger.info("classify_intent → intents=%s (query_len=%d)", intents, len(query))
    return {"intents": intents}


def handle_unknown(state: GraphState) -> GraphState:
    """Respuesta estándar para consultas fuera de alcance."""
    return {
        "final_answer": (
            "No encontré información suficiente en la base documental proporcionada "
            "para responder esta consulta. Mi conocimiento está limitado a los documentos "
            "de Arquitectura de Microservicios, Lineamientos de Seguridad para APIs y "
            "Checklist de Paso a Producción del banco."
        ),
        "agents_invoked": [],
        "sources": [],
        "warnings": [
            "La consulta no corresponde a ningún dominio cubierto por la base documental."
        ],
    }


def run_agent_node(state: GraphState) -> GraphState:
    """
    Nodo de agente individual. Recibe el intent en state["intents"][0]
    (LangGraph Send API envía el intent como parte del estado).
    """
    # Cuando se usa Send, el estado contiene el intent como primer elemento
    intent = state["intents"][0]
    query = state["query"]

    agent_class = AGENT_REGISTRY.get(intent)
    if not agent_class:
        return {"agent_responses": []}

    agent = agent_class()
    logger.info("Ejecutando %s", agent.agent_name)
    response = agent.run(query)
    return {"agent_responses": [response]}


def consolidate(state: GraphState) -> GraphState:
    """
    Consolida las respuestas parciales de todos los agentes en una respuesta final.
    Detecta si todos los agentes fallaron y agrega advertencia.
    """
    responses: list[AgentResponse] = state["agent_responses"]
    query = state["query"]

    all_no_info = all(r.no_info_flag for r in responses)

    if all_no_info:
        return {
            "final_answer": (
                "No encontré información suficiente en la base documental proporcionada "
                "para responder esta consulta."
            ),
            "agents_invoked": [r.agent_name for r in responses],
            "sources": [],
            "warnings": [
                "Ningún agente encontró información relevante en la base documental. "
                "Verifique que la consulta esté dentro del alcance del sistema."
            ],
        }

    # Construir contexto de respuestas parciales para consolidación
    partial_texts = []
    for r in responses:
        if not r.no_info_flag:
            partial_texts.append(
                f"### {r.agent_name}\n{r.partial_answer}"
            )

    combined = "\n\n".join(partial_texts)

    llm = get_llm()

    system = (
        "Eres el consolidador de respuestas de una mesa de ayuda técnica bancaria. "
        "Recibirás respuestas parciales de distintos agentes especializados. "
        "Tu tarea es unificarlas en una respuesta coherente, clara y estructurada "
        "para el usuario final. "
        "Mantén toda la información relevante, elimina redundancias y usa encabezados "
        "cuando haya respuestas de múltiples dominios. "
        "No agregues información que no esté en las respuestas parciales. "
        "Responde en español."
    )

    user_msg = (
        f"Pregunta original del usuario:\n{query}\n\n"
        f"Respuestas parciales de los agentes:\n\n{combined}"
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])

    # Consolidar fuentes (deduplicar por section_title + source_file)
    seen = set()
    all_sources = []
    for r in responses:
        for s in r.sources:
            key = (s["source_file"], s["section_title"])
            if key not in seen:
                seen.add(key)
                all_sources.append(s)

    # Ordenar fuentes por score descendente
    all_sources.sort(key=lambda x: x["score"], reverse=True)

    # Advertencias si algún agente no encontró info
    warnings = []
    for r in responses:
        if r.no_info_flag:
            warnings.append(
                f"{r.agent_name}: no encontró información suficiente en su dominio."
            )

    return {
        "final_answer": response.content,
        "agents_invoked": [r.agent_name for r in responses],
        "sources": all_sources,
        "warnings": warnings,
    }


# ─── Routing condicional ──────────────────────────────────────────────────────

def route_after_classify(state: GraphState):
    """
    Decide qué nodos ejecutar según los intents clasificados.
    Usa la Send API de LangGraph para fan-out paralelo a cada agente.
    """
    intents = state["intents"]

    if "unknown" in intents or not intents:
        return "handle_unknown"

    # Fan-out: enviar una tarea por cada intent al nodo run_agent_node
    return [
        Send("run_agent_node", {**state, "intents": [intent]})
        for intent in intents
        if intent in AGENT_REGISTRY
    ]


# ─── Construcción del grafo ───────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("run_agent_node", run_agent_node)
    graph.add_node("consolidate", consolidate)
    graph.add_node("handle_unknown", handle_unknown)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
    )

    graph.add_edge("run_agent_node", "consolidate")
    graph.add_edge("consolidate", END)
    graph.add_edge("handle_unknown", END)

    return graph.compile()


# Instancia compilada — se reutiliza en toda la app
orchestrator = build_graph()
