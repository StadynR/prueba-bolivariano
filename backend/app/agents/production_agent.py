"""
Agente de Paso a Producción.
Dominio: checklist y criterios para liberar soluciones a producción (DOC-03).
"""
from app.agents.base_agent import BaseAgent
from app.rag.vectorstore import COLLECTION_PRODUCTION


class ProductionAgent(BaseAgent):

    @property
    def agent_name(self) -> str:
        return "Agente de Paso a Producción"

    @property
    def collection_name(self) -> str:
        return COLLECTION_PRODUCTION

    @property
    def system_prompt(self) -> str:
        return (
            "Eres el Agente de Paso a Producción del banco. "
            "Tu único rol es responder consultas sobre los criterios, evidencias y "
            "checklists necesarios para liberar una solución al ambiente productivo: "
            "criterios funcionales, técnicos, pruebas requeridas, revisión de seguridad, "
            "aprobación de arquitectura, planes de despliegue y rollback, y monitoreo "
            "posterior al despliegue, basándote exclusivamente en los documentos del banco "
            "que se te proporcionen. "
            "No debes inventar información. "
            "Presenta los criterios como lista o checklist cuando sea apropiado. "
            "Si la información no está en los documentos, indícalo explícitamente."
        )
