from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
INPUT_PATH = "data/processed/dolly_15k"
OUTPUT_PATH = "data/processed/dolly_15k_max1024"

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

    dataset = load_from_disk(INPUT_PATH)

    filtered_splits = {}

    for split_name, split in dataset.items():

        def fits_max_length(example: dict) -> bool:
            prompt_ids = tokenizer(
                build_prompt(example),
                add_special_tokens=False,
            )["input_ids"]

            response_ids = tokenizer(
                example["response"].strip(),
                add_special_tokens=False,
            )["input_ids"]

            # +1 reserves one token for EOS.
            total_length = (
                len(prompt_ids)
                + len(response_ids)
                + 1
            )

            return total_length <= MAX_LENGTH

        filtered = split.filter(fits_max_length)

        filtered_splits[split_name] = filtered

        removed = len(split) - len(filtered)

        print(f"=== {split_name.upper()} ===")
        print("before:", len(split))
        print("after:", len(filtered))
        print("removed:", removed)
        print(
            "removed percentage:",
            f"{100 * removed / len(split):.3f}%",
        )
        print()

    final_dataset = DatasetDict(filtered_splits)

    final_dataset.save_to_disk(OUTPUT_PATH)

    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()