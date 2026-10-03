"""
State definition for Enterprise Agentic RAG
Defines typed contracts for LangGraph node transitions.
"""

from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_name: str
    page_number: int
    content: str
    is_table: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0


class AgenticRAGState(BaseModel):
    """
    Durable graph state checkpointed at each step.
    """
    user_query: str
    sub_queries: List[str] = Field(default_factory=list)
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    critic_notes: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    iteration_count: int = 0
    max_iterations: int = 2
    requires_human_review: bool = False
    human_approved: bool = False
    final_response: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
