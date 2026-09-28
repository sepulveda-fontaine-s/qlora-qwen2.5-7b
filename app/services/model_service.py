from abc import ABC, abstractmethod

from app.schemas.generation import GenerationRequest, GenerationResponse


class ModelService(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise NotImplementedError