import gc
import math
import argparse
import torch
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_PATH = "data/raw/models/Qwen2.5-7B"

DATASET_PATH = "data/processed/dolly_15k_tokenized"

BATCH_SIZE = 4


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


def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_target_tokens = 0

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            # Causal LM loss predicts token t from tokens before t,
            # therefore labels are internally shifted by one position.
            target_tokens = (
                batch["labels"][:, 1:] != -100
            ).sum().item()

            total_loss += outputs.loss.item() * target_tokens
            total_target_tokens += target_tokens

    mean_loss = total_loss / total_target_tokens
    perplexity = math.exp(mean_loss)

    return mean_loss, perplexity


def load_base_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the LoRA/QLoRA adapter.",
    )

    parser.add_argument(
        "--label",
        required=True,
        help="Label used in the evaluation output.",
    )

    args = parser.parse_args()

    adapter_path = args.adapter
    label = args.label
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    validation = load_from_disk(DATASET_PATH)["validation"]

    loader = DataLoader(
        validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda examples: collate_batch(
            examples,
            tokenizer,
        ),
        num_workers=2,
        pin_memory=True,
    )

    print("=== VALIDATION SET ===")
    print("examples:", len(validation))

    print("\n=== BASE MODEL ===")

    base_model = load_base_model()

    base_loss, base_perplexity = evaluate(
        base_model,
        loader,
    )

    print(f"validation loss: {base_loss:.6f}")
    print(f"perplexity: {base_perplexity:.4f}")

    print("\n=== LOAD QLORA ADAPTER ===")

    adapted_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )

    adapted_loss, adapted_perplexity = evaluate(
        adapted_model,
        loader,
    )

    print(f"\n=== {label} ===")
    print(f"validation loss: {adapted_loss:.6f}")
    print(f"perplexity: {adapted_perplexity:.4f}")

    print("\n=== COMPARISON ===")
    print(
        "loss difference:",
        round(adapted_loss - base_loss, 6),
    )
    print(
        "loss change %:",
        round(
            100 * (adapted_loss - base_loss) / base_loss,
            2,
        ),
    )

    print("\n=== RESULT ===")
    print("Base vs QLoRA validation evaluation: SUCCESS")


if __name__ == "__main__":
    main()