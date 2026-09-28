import os


class Settings:
    app_name: str = os.getenv("APP_NAME", "LoRA / QLoRA Production API")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")

    model_mode: str = os.getenv("MODEL_MODE", "mock")

    model_path: str = os.getenv(
        "MODEL_PATH",
        "data/raw/models/Qwen2.5-7B",
    )

    adapter_path: str = os.getenv(
        "ADAPTER_PATH",
        "results/qlora_r8/final_adapter",
    )

    max_new_tokens: int = int(
        os.getenv("MAX_NEW_TOKENS", "256")
    )

    def validate(self) -> None:
        allowed_modes = {
            "mock",
            "qlora_4bit_peft",
        }

        if self.model_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported MODEL_MODE: {self.model_mode}. "
                f"Allowed values: {sorted(allowed_modes)}"
            )


settings = Settings()
settings.validate()