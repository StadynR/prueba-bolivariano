"""
Agente de Seguridad y Cumplimiento.
Dominio: lineamientos de seguridad para APIs y soluciones IA (DOC-02).
"""
from app.agents.base_agent import BaseAgent
from app.rag.vectorstore import COLLECTION_SECURITY


class SecurityAgent(BaseAgent):

    @property
    def agent_name(self) -> str:
        return "Agente de Seguridad"

    @property
    def collection_name(self) -> str:
        return COLLECTION_SECURITY

    @property
    def system_prompt(self) -> str:
        return (
            "Eres el Agente de Seguridad y Cumplimiento del banco. "
            "Tu único rol es responder consultas sobre controles de seguridad, "
            "autenticación, autorización, manejo de secretos, protección de información "
            "sensible, auditoría, guardrails en soluciones IA y cumplimiento normativo, "
            "basándote exclusivamente en los documentos del banco que se te proporcionen. "
            "No debes inventar información ni usar conocimiento externo a los documentos. "
            "Cuando identifiques un riesgo de seguridad en la consulta, señálalo explícitamente. "
            "Si la información no está en los documentos, indícalo sin suponer controles."
        )
