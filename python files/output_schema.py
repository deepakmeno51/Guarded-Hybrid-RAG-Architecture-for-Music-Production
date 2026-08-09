from pydantic import BaseModel, Field, field_validator


class RAGResponse(BaseModel):
    answer: str
    sources: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v):
        if not v.strip():
            raise ValueError("empty answer")
        return v

    @field_validator("sources")
    @classmethod
    def sources_valid(cls, v):
        if any(not s.strip() for s in v):
            raise ValueError("empty source entry")
        return v
