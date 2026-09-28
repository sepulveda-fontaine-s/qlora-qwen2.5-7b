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

BATCH_SIZE = 4
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

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    dataset = load_from_disk(DATASET_PATH)["train"]

    # Select the two longest examples in the dataset.
    sorted_indices = sorted(
        range(len(dataset)),
        key=lambda i: len(dataset[i]["input_ids"]),
        reverse=True,
    )

    selected = [
        dataset[i]
        for i in sorted_indices[:BATCH_SIZE]
    ]

    print("=== SELECTED LENGTHS ===")
    print([len(example["input_ids"]) for example in selected])

    batch = collate_batch(selected, tokenizer)

    batch = {
        key: value.cuda()
        for key, value in batch.items()
    }

    optimizer.zero_grad()

    outputs = model(**batch)
    loss = outputs.loss

    print("\n=== LOSS ===")
    print(loss.item())

    loss.backward()
    optimizer.step()

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
    print("QLoRA batch_size=2 worst-case stress test: SUCCESS")


if __name__ == "__main__":
    main()