from time import perf_counter

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from app.core.config import settings
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.services.model_service import ModelService


class QLoRAModelService(ModelService):
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for qlora_4bit_peft mode."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            local_files_only=True,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            settings.model_path,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            settings.adapter_path,
            is_trainable=False,
        )

        self.model.eval()

    @staticmethod
    def _build_prompt(
        instruction: str,
        context: str | None,
    ) -> str:
        instruction = instruction.strip()
        context = (context or "").strip()

        if context:
            return (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Context:\n"
                f"{context}\n\n"
                "### Response:\n"
            )

        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        prompt = self._build_prompt(
            request.instruction,
            request.context,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

        start = perf_counter()

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency = perf_counter() - start

        generated_tokens = generated[0][input_length:]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return GenerationResponse(
            text=text,
            model_mode="qlora_4bit_peft",
            latency_seconds=latency,
            generated_tokens=len(generated_tokens),
        )