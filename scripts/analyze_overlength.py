from collections import Counter

from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
DATASET_PATH = "data/processed/dolly_15k"
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

    dataset = load_from_disk(DATASET_PATH)["train"]

    category_names = dataset.features["category"].names

    total_overlength = Counter()
    prompt_overlength = Counter()

    for example in dataset:
        category = category_names[example["category"]]

        prompt_ids = tokenizer(
            build_prompt(example),
            add_special_tokens=False,
        )["input_ids"]

        response_ids = tokenizer(
            example["response"].strip(),
            add_special_tokens=False,
        )["input_ids"]

        total_length = len(prompt_ids) + len(response_ids) + 1

        if total_length > MAX_LENGTH:
            total_overlength[category] += 1

        if len(prompt_ids) >= MAX_LENGTH:
            prompt_overlength[category] += 1

    print("=== TOTAL > 1024 BY CATEGORY ===")
    for category, count in sorted(total_overlength.items()):
        print(f"{category}: {count}")

    print("\n=== PROMPT >= 1024 BY CATEGORY ===")
    for category, count in sorted(prompt_overlength.items()):
        print(f"{category}: {count}")

    print("\n=== TOTAL ===")
    print("total > 1024:", sum(total_overlength.values()))
    print("prompt >= 1024:", sum(prompt_overlength.values()))


if __name__ == "__main__":
    main()