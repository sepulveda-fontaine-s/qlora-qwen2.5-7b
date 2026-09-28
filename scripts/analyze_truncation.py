import numpy as np
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

    prompt_lengths = []
    response_lengths = []
    total_lengths = []

    prompt_over_limit = 0
    total_over_limit = 0

    for example in dataset:
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

        # +1 for EOS token at the end of the response.
        total_length = len(prompt_ids) + len(response_ids) + 1

        prompt_lengths.append(len(prompt_ids))
        response_lengths.append(len(response_ids))
        total_lengths.append(total_length)

        if len(prompt_ids) >= MAX_LENGTH:
            prompt_over_limit += 1

        if total_length > MAX_LENGTH:
            total_over_limit += 1

    print("=== TRUNCATION ANALYSIS ===")
    print("examples:", len(dataset))
    print("max_length candidate:", MAX_LENGTH)

    print("\n=== PROMPT ===")
    print("median:", int(np.median(prompt_lengths)))
    print("p95:", int(np.percentile(prompt_lengths, 95)))
    print("p99:", int(np.percentile(prompt_lengths, 99)))
    print("max:", max(prompt_lengths))

    print("\n=== RESPONSE ===")
    print("median:", int(np.median(response_lengths)))
    print("p95:", int(np.percentile(response_lengths, 95)))
    print("p99:", int(np.percentile(response_lengths, 99)))
    print("max:", max(response_lengths))

    print("\n=== IMPACT AT 1024 ===")
    print(
        "prompt >= 1024:",
        prompt_over_limit,
        f"({100 * prompt_over_limit / len(dataset):.3f}%)",
    )
    print(
        "total > 1024:",
        total_over_limit,
        f"({100 * total_over_limit / len(dataset):.3f}%)",
    )


if __name__ == "__main__":
    main()