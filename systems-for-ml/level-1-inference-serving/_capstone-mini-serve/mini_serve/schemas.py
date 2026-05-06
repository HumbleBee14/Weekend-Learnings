"""Pydantic request/response models — single source of truth for the API contract."""

from typing import Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


class GenerateResponse(BaseModel):
    completion: str
    tokens_generated: int
    request_id: str
    metrics: "RequestMetrics"


class RequestMetrics(BaseModel):
    queue_wait_ms: float
    ttft_ms: Optional[float] = None  # only set on streaming
    decode_ms: float
    total_ms: float
    batch_size_seen: int


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    queue_depth: int


GenerateResponse.model_rebuild()  # resolve forward refs
