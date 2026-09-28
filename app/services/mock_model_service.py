from time import perf_counter

from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.model_service import ModelService


class MockModelService(ModelService):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start = perf_counter()

        text = f"Mock response for: {request.instruction}"

        latency = perf_counter() - start

        return GenerationResponse(
            text=text,
            model_mode="mock",
            latency_seconds=latency,
            generated_tokens=len(text.split()),
        )