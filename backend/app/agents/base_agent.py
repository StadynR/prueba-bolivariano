"""
Clase base abstracta para todos los agentes especializados.

Diseño:
  - Cada agente es responsable de UNA colección ChromaDB (su dominio).
  - Recibe la query, recupera chunks relevantes, genera una respuesta parcial.
  - Activa no_info_flag si no hay chunks sobre el umbral → evita alucinaciones.
  - El orquestador consolida las respuestas parciales de todos los agentes.

Para agregar un nuevo agente:
  1. Crear una nueva clase que herede de BaseAgent.
  2. Definir collection_name, agent_name y system_prompt.
  3. Registrarlo en el orquestador.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from app.rag.retriever import retrieve, RetrievedChunk
from app.config import settings
from app.llm_factory import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


@dataclass
class AgentResponse:
    agent_name: str
    partial_answer: str
    sources: list[dict]          # [{source_file, section_title, doc_id, score}]
    no_info_flag: bool = False    # True = no se encontró información suficiente
    tokens_used: int = 0


class BaseAgent(ABC):
    """Agente especializado base. Cada subclase define su dominio."""

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Nombre legible del agente."""

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Nombre de la colección ChromaDB que este agente puede consultar."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Instrucción del sistema que ancla al agente a su dominio."""

    def run(self, query: str) -> AgentResponse:
        """
        Ejecuta el agente:
          1. Recupera chunks relevantes de su colección.
          2. Si no hay chunks suficientes → respuesta estándar sin inventar.
          3. Si hay contexto → genera respuesta con LLM usando solo ese contexto.
        """
        chunks: list[RetrievedChunk] = retrieve(query, self.collection_name)

        if not chunks:
            return AgentResponse(
                agent_name=self.agent_name,
                partial_answer=(
                    "No encontré información suficiente en la base documental "
                    "proporcionada para responder esta consulta desde mi área."
                ),
                sources=[],
                no_info_flag=True,
            )

        context = self._build_context(chunks)
        prompt = self._build_user_prompt(query, context)

        llm = get_llm()

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)
        answer = response.content

        # Si el LLM no pudo responder con el contexto, activar flag
        no_info = self._detect_no_info(answer)

        sources = [
            {
                "source_file": c.source_file,
                "section_title": c.section_title,
                "doc_id": c.doc_id,
                "score": c.score,
            }
            for c in chunks
        ]

        tokens_used = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens_used = response.usage_metadata.get("total_tokens", 0)

        return AgentResponse(
            agent_name=self.agent_name,
            partial_answer=answer,
            sources=sources,
            no_info_flag=no_info,
            tokens_used=tokens_used,
        )

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Fragmento {i} - {chunk.section_title} ({chunk.source_file})]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            f"A continuación se presentan fragmentos del documento de referencia "
            f"para tu área:\n\n{context}\n\n"
            f"Basándote ÚNICAMENTE en los fragmentos anteriores, responde la siguiente "
            f"consulta de manera clara y estructurada:\n\n{query}\n\n"
            f"Si los fragmentos no contienen información suficiente para responder, "
            f'responde exactamente: "No encontré información suficiente en la base '
            f'documental proporcionada."'
        )

    def _detect_no_info(self, answer: str) -> bool:
        return "no encontré información suficiente" in answer.lower()
