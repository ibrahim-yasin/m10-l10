"""Pydantic request/response models for the recipe service.

These are the typed-boundary contracts. They must mirror the TypeScript
interfaces in `web/lib/types.ts` exactly — drift produces silent render
failures in the Next.js frontend.
"""
from typing import List, Literal

from pydantic import BaseModel, Field


# --- /extract --------------------------------------------------------

class ExtractRequest(BaseModel):
    """Request body for POST /extract.

    The request field has a length constraint that gates 422 on empty
    or oversized input.
    """
    # TODO: declare the request body field with a Field(...) length
    #       constraint.
    text: str = Field(..., min_length=1, max_length=1000)   



class Entity(BaseModel):
    """A single named-entity span.

    Field names must match the corresponding TypeScript Entity
    interface in `web/lib/types.ts` exactly.
    """
    # TODO: declare the span's text, label, and start/end character
    #       offsets.
    text: str
    label: str
    start: int
    end: int


class ExtractResponse(BaseModel):
    """Response body for POST /extract.

    Per the Evaluation Methodology, the returned list is ordered by
    start offset ascending.
    """
    # TODO: declare the field that carries the ordered list of entities.
    entities: list[Entity]  


# --- /kg/query -------------------------------------------------------

class KGRequest(BaseModel):
    """Request body for POST /kg/query.

    The question field has a length constraint.
    """
    # TODO: declare the request field with a Field(...) length
    #       constraint.
    question: str = Field(..., min_length=1, max_length=500)    



class KGResponse(BaseModel):
    """Response body for POST /kg/query."""
    # TODO: declare the cypher string, the rows the driver returned,
    #       and the row count.
    cypher: str
    rows: list[dict]
    count: int


class UnsupportedQueryDetail(BaseModel):
    """Structured detail returned on 422 from /kg/query."""
    reason: Literal["unsupported_question"]
    supported_patterns: list[str]


# --- /rag/answer -----------------------------------------------------

class RAGRequest(BaseModel):
    """Request body for POST /rag/answer.

    The question field has a length constraint; `k` is a bounded
    integer with a default.
    """
    # TODO: declare the question field with a Field(...) length
    #       constraint and a bounded integer `k` with a default.
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(5, ge=1, le=10)


class Citation(BaseModel):
    """One citation: chunk id and retrieval score.

    Field names must match the TypeScript Citation interface.
    """
    # TODO: declare the citation's chunk identifier and retrieval score.
    chunk_id: str
    score: float    


class RAGResponse(BaseModel):
    """Response body for POST /rag/answer.

    Grounding contract: when `answer` is not the empty-retrieval
    sentinel, `len(citations) > 0` is required.
    """
    # TODO: declare the answer string, the list of citations, and the
    #       confidence score.
    answer: str
    citations: list[Citation]
    confidence: float


# --- Health / readiness ---------------------------------------------

class HealthResponse(BaseModel):
    """Liveness response."""
    # TODO: declare the single field returned by /healthz.
    status: str = "ok"  


class ReadyDetail(BaseModel):
    """Readiness detail naming each backend's status."""
    neo4j: str
    weaviate: str
