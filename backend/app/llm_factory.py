"""
Clientes LLM y Embeddings usando OpenAI vía LangChain.

get_llm()        → ChatOpenAI singleton (usado por agentes y orquestador)
get_embeddings() → OpenAIEmbeddings singleton (usado por retriever e ingest)

Los modelos se configuran con LLM_MODEL y EMBEDDING_MODEL en .env.
"""
from __future__ import annotations
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Devuelve el cliente ChatOpenAI configurado (singleton por proceso)."""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Devuelve el cliente OpenAIEmbeddings configurado (singleton por proceso)."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
