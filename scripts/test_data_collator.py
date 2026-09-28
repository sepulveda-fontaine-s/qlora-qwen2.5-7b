import torch
from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
DATASET_PATH = "data/processed/dolly_15k_tokenized"


def collate_batch(examples, tokenizer):
    max_length = max(len(example["input_ids"]) for example in examples)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for example in examples:
        length = len(example["input_ids"])
        padding_length = max_length - length

        padded_input_ids.append(
            example["input_ids"]
            + [tokenizer.pad_token_id] * padding_length
        )

        padded_attention_mask.append(
            example["attention_mask"]
            + [0] * padding_length
        )

        # Padding must never contribute to the loss.
        padded_labels.append(
            example["labels"]
            + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            padded_attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    dataset = load_from_disk(DATASET_PATH)["train"]

    # Deliberately use examples with different lengths.
    examples = [
        dataset[0],
        dataset[1],
        dataset[2],
        dataset[3],
    ]

    original_lengths = [
        len(example["input_ids"])
        for example in examples
    ]

    batch = collate_batch(examples, tokenizer)

    print("=== ORIGINAL LENGTHS ===")
    print(original_lengths)

    print("\n=== BATCH SHAPES ===")
    for key, tensor in batch.items():
        print(key, tuple(tensor.shape))

    print("\n=== PADDING CHECK ===")
    for index, original_length in enumerate(original_lengths):
        padded_positions = batch["attention_mask"][index] == 0

        padding_tokens = int(padded_positions.sum())

        labels_correct = bool(
            torch.all(
                batch["labels"][index][padded_positions] == -100
            )
        )

        print(
            f"example {index}: "
            f"original={original_length}, "
            f"padding={padding_tokens}, "
            f"padding_labels_are_-100={labels_correct}"
        )


if __name__ == "__main__":
    main()