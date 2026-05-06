"""
Modelos de datos compartidos entre orquestador, agentes y API.
"""
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class SourceReference(BaseModel):
    source_file: str
    section_title: str
    doc_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    agents_invoked: list[str]
    sources: list[SourceReference]
    warnings: list[str]
    query_id: str
