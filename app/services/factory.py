from app.core.config import settings
from app.services.mock_model_service import MockModelService


def get_model_service():
    if settings.model_mode == "mock":
        return MockModelService()

    if settings.model_mode == "qlora_4bit_peft":
        from app.services.qlora_model_service import QLoRAModelService

        return QLoRAModelService()

    raise ValueError(f"Unsupported MODEL_MODE: {settings.model_mode}")