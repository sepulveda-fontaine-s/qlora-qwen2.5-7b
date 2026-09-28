from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    context: str | None = None


class GenerationResponse(BaseModel):
    text: str
    model_mode: str
    latency_seconds: float
    generated_tokens: int