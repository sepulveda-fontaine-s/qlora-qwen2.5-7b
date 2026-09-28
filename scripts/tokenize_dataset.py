from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
INPUT_PATH = "data/processed/dolly_15k_max1024"
OUTPUT_PATH = "data/processed/dolly_15k_tokenized"

MAX_LENGTH = 1024


def build_prompt(example: dict) -> str:
    instruction = example["instruction"].strip()
    context = example["context"].strip()

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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")

    dataset = load_from_disk(INPUT_PATH)

    def tokenize_example(example: dict) -> dict:
        prompt = build_prompt(example)
        response = example["response"].strip()

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = (
            prompt_ids
            + response_ids
            + [tokenizer.eos_token_id]
        )

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Example exceeds MAX_LENGTH: {len(input_ids)}"
            )

        # Ignore prompt tokens in the training loss.
        labels = (
            [-100] * len(prompt_ids)
            + response_ids
            + [tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    tokenized_splits = {}

    for split_name, split in dataset.items():
        tokenized = split.map(
            tokenize_example,
            remove_columns=split.column_names,
            desc=f"Tokenizing {split_name}",
        )

        tokenized_splits[split_name] = tokenized

        lengths = [
            len(example["input_ids"])
            for example in tokenized
        ]

        print(f"=== {split_name.upper()} ===")
        print("examples:", len(tokenized))
        print("max tokens:", max(lengths))
        print("min tokens:", min(lengths))

        first = tokenized[0]

        trainable_labels = sum(
            token != -100
            for token in first["labels"]
        )

        ignored_labels = sum(
            token == -100
            for token in first["labels"]
        )

        print("first example ignored prompt tokens:", ignored_labels)
        print("first example trained response tokens:", trainable_labels)
        print()

    final_dataset = DatasetDict(tokenized_splits)

    final_dataset.save_to_disk(OUTPUT_PATH)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()