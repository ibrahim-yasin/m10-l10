from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Entity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    label: str
    start: int = Field(validation_alias=AliasChoices("start", "start_char"))
    end: int = Field(validation_alias=AliasChoices("end", "end_char"))

    @property
    def start_char(self) -> int:
        return self.start

    @property
    def end_char(self) -> int:
        return self.end


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ExtractResponse(BaseModel):
    entities: list[Entity]


class KGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class KGResponse(BaseModel):
    cypher: str
    rows: list[dict[str, Any]]
    count: int


class Citation(BaseModel):
    chunk_id: int
    score: float


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float


class HealthResponse(BaseModel):
    status: str


class UnsupportedQueryDetail(BaseModel):
    reason: str
    supported_patterns: list[str]