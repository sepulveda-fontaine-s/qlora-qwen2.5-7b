from collections import Counter

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-7B"
CACHE_DIR = "data/raw/hf_cache"


def main() -> None:
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    # Build only the model structure.
    # Parameters live on the "meta" device: no real model weights are allocated.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    linear_modules = []

    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf_name = full_name.split(".")[-1]

            linear_modules.append(
                (
                    full_name,
                    leaf_name,
                    module.in_features,
                    module.out_features,
                )
            )

    counts = Counter(item[1] for item in linear_modules)

    print("=== UNIQUE LINEAR MODULE NAMES ===")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")

    print("\n=== FIRST TRANSFORMER BLOCK ===")
    for full_name, leaf_name, in_features, out_features in linear_modules:
        if full_name.startswith("model.layers.0."):
            print(
                f"{full_name}: "
                f"{in_features} -> {out_features}"
            )

    print("\n=== OUTPUT HEAD ===")
    for full_name, _, in_features, out_features in linear_modules:
        if full_name == "lm_head":
            print(
                f"{full_name}: "
                f"{in_features} -> {out_features}"
            )


if __name__ == "__main__":
    main()