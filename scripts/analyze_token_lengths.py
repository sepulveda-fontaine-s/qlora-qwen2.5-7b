from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
DATASET_PATH = "data/processed/dolly_15k"


def format_example(example: dict) -> str:
    instruction = example["instruction"].strip()
    context = example["context"].strip()
    response = example["response"].strip()

    if context:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return prompt + response


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    dataset = load_from_disk(DATASET_PATH)

    lengths = []

    for example in dataset["train"]:
        text = format_example(example)

        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        lengths.append(len(token_ids))

    values = np.array(lengths)

    print("=== TOKEN LENGTH DISTRIBUTION ===")
    print("examples:", len(values))
    print("mean:", round(values.mean(), 2))
    print("median:", int(np.percentile(values, 50)))
    print("p90:", int(np.percentile(values, 90)))
    print("p95:", int(np.percentile(values, 95)))
    print("p99:", int(np.percentile(values, 99)))
    print("max:", int(values.max()))

    for limit in [256, 512, 1024, 2048]:
        pct = 100 * np.mean(values > limit)
        print(f"> {limit}: {pct:.2f}%")


if __name__ == "__main__":
    main()