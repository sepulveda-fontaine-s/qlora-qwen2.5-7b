import random

import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
DATASET_PATH = "data/processed/dolly_15k_tokenized"

SEED = 42
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
OPTIMIZER_STEPS = 3
LEARNING_RATE = 2e-4


def collate_batch(examples, tokenizer):
    max_length = max(len(example["input_ids"]) for example in examples)

    input_ids = []
    attention_mask = []
    labels = []

    for example in examples:
        padding = max_length - len(example["input_ids"])

        input_ids.append(
            example["input_ids"]
            + [tokenizer.pad_token_id] * padding
        )

        attention_mask.append(
            example["attention_mask"]
            + [0] * padding
        )

        labels.append(
            example["labels"]
            + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("=== LOAD MODEL ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

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

    model = get_peft_model(model, lora_config)
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    dataset = load_from_disk(DATASET_PATH)["train"]

    # Small reproducible subset for the smoke test.
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    required_examples = (
        OPTIMIZER_STEPS
        * GRADIENT_ACCUMULATION_STEPS
        * BATCH_SIZE
    )

    indices = indices[:required_examples]

    print("\n=== TRAINING CONFIG ===")
    print("examples:", required_examples)
    print("batch size:", BATCH_SIZE)
    print(
        "gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )
    print("optimizer steps:", OPTIMIZER_STEPS)
    print("learning rate:", LEARNING_RATE)

    optimizer.zero_grad()

    running_loss = 0.0
    optimizer_step = 0

    for micro_step, index in enumerate(indices, start=1):
        example = dataset[index]

        batch = collate_batch(
            [example],
            tokenizer,
        )

        batch = {
            key: value.cuda()
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss
        scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        scaled_loss.backward()

        running_loss += loss.item()

        if micro_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            optimizer_step += 1

            average_loss = (
                running_loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            print(
                f"optimizer_step={optimizer_step} "
                f"average_loss={average_loss:.4f}"
            )

            running_loss = 0.0

    print("\n=== GPU MEMORY ===")
    print(
        "allocated GiB:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
    )
    print(
        "reserved GiB:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
    )
    print(
        "peak allocated GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    )

    print("\n=== RESULT ===")
    print("QLoRA training smoke test: SUCCESS")


if __name__ == "__main__":
    main()