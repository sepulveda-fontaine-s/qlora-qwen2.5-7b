import torch

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


MODEL_PATH = "data/raw/models/Qwen2.5-7B"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("=== LOAD 4-BIT BASE MODEL ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    # Prepare a quantized model for parameter-efficient training.
    # The base model remains frozen.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    print("\n=== ATTACH LORA ADAPTER ===")

    model = get_peft_model(
        model,
        lora_config,
    )

    total_params = 0
    trainable_params = 0
    unexpected_trainable = []

    for name, parameter in model.named_parameters():
        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()

            if "lora_" not in name:
                unexpected_trainable.append(name)

    percentage = 100 * trainable_params / total_params

    print("\n=== TRAINABLE PARAMETERS ===")
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    print(f"trainable percentage: {percentage:.4f}%")

    print("\n=== FREEZE CHECK ===")

    if unexpected_trainable:
        print("Unexpected trainable parameters:")
        for name in unexpected_trainable:
            print(name)
    else:
        print("PASS: only LoRA parameters are trainable.")

    print("\n=== LORA CONFIG ===")
    print("r:", lora_config.r)
    print("alpha:", lora_config.lora_alpha)
    print("dropout:", lora_config.lora_dropout)
    print("scaling alpha/r:", lora_config.lora_alpha / lora_config.r)
    print("targets:", lora_config.target_modules)

    print("\n=== GPU MEMORY ===")
    print(
        "allocated GiB:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
    )
    print(
        "reserved GiB:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
    )


if __name__ == "__main__":
    main()