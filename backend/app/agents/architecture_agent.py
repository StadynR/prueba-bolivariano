"""
Agente de Arquitectura Técnica.
Dominio: estándares de arquitectura y microservicios (DOC-01).
"""
from app.agents.base_agent import BaseAgent
from app.rag.vectorstore import COLLECTION_ARCHITECTURE


class ArchitectureAgent(BaseAgent):

    @property
    def agent_name(self) -> str:
        return "Agente de Arquitectura"

    @property
    def collection_name(self) -> str:
        return COLLECTION_ARCHITECTURE

    @property
    def system_prompt(self) -> str:
        return (
            "Eres el Agente de Arquitectura Técnica del banco. "
            "Tu único rol es responder consultas sobre estándares de arquitectura, "
            "diseño de microservicios, APIs internas, health checks, trazabilidad, "
            "manejo de errores, pruebas técnicas y observabilidad, "
            "basándote exclusivamente en los documentos del banco que se te proporcionen. "
            "No debes inventar información ni usar conocimiento externo a los documentos. "
            "Cuando cites criterios, menciona la sección de la que provienen. "
            "Si la información no está en los documentos, indícalo explícitamente."
        )
