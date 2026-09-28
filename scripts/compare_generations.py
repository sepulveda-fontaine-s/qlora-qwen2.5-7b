import gc
import json
import random
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
DATASET_PATH = "data/processed/dolly_15k_max1024"

ADAPTERS = {
    "base": None,
    "qlora_r8": "results/qlora_r8/final_adapter",
    "qlora_r16": "results/qlora_r16/final_adapter",
}

OUTPUT_PATH = Path(
    "results/generation_comparison/base_vs_r8_vs_r16.jsonl"
)

SEED = 42
EXAMPLES_PER_CATEGORY = 3
MAX_NEW_TOKENS = 256


def build_prompt(example):
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


def select_examples(dataset):
    category_names = dataset.features["category"].names

    grouped = {name: [] for name in category_names}

    for index, example in enumerate(dataset):
        category = category_names[example["category"]]
        grouped[category].append(index)

    rng = random.Random(SEED)

    selected = []

    for category in category_names:
        indices = grouped[category]
        chosen = rng.sample(
            indices,
            EXAMPLES_PER_CATEGORY,
        )

        for index in chosen:
            example = dataset[index]

            selected.append(
                {
                    "dataset_index": index,
                    "category": category,
                    "instruction": example["instruction"],
                    "context": example["context"],
                    "reference": example["response"],
                    "prompt": build_prompt(example),
                }
            )

    return selected


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


def generate_outputs(label, adapter_path, examples, tokenizer):
    print(f"\n=== {label.upper()} ===")

    model = load_base_model()

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()

    outputs = []

    for number, example in enumerate(examples, start=1):
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_tokens = generated[0][input_length:]

        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        outputs.append(text)

        print(
            f"{number}/{len(examples)} "
            f"category={example['category']}"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return outputs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    validation = load_from_disk(
        DATASET_PATH
    )["validation"]

    examples = select_examples(validation)

    print("=== EVALUATION SAMPLE ===")
    print("examples:", len(examples))
    print(
        "categories:",
        sorted(set(x["category"] for x in examples)),
    )

    generations = {}

    for label, adapter_path in ADAPTERS.items():
        generations[label] = generate_outputs(
            label,
            adapter_path,
            examples,
            tokenizer,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for index, example in enumerate(examples):
            record = {
                **example,
                "base": generations["base"][index],
                "qlora_r8": generations["qlora_r8"][index],
                "qlora_r16": generations["qlora_r16"][index],
            }

            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print("\n=== RESULT ===")
    print("Saved:", OUTPUT_PATH)
    print("Generation comparison: SUCCESS")


if __name__ == "__main__":
    main()