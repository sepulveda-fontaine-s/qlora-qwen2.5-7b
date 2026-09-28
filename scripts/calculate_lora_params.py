from collections import defaultdict

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-7B"
CACHE_DIR = "data/raw/hf_cache"

RANKS = [8, 16, 32]

TARGET_SETS = {
    "q_v_only": {
        "q_proj",
        "v_proj",
    },
    "attention_all": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    },
    "all_transformer_linear": {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    },
}


def main() -> None:
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    total_base_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    linear_modules = []

    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            leaf_name = full_name.split(".")[-1]

            # Exclude the language-model output head.
            if leaf_name == "lm_head":
                continue

            linear_modules.append(
                (
                    full_name,
                    leaf_name,
                    module.in_features,
                    module.out_features,
                )
            )

    print("=== BASE MODEL ===")
    print(f"parameters: {total_base_params:,}")

    print("\n=== LORA PARAMETER COUNTS ===")

    for target_name, targets in TARGET_SETS.items():
        print(f"\n[{target_name}]")

        selected = [
            module
            for module in linear_modules
            if module[1] in targets
        ]

        print("target modules:", sorted(targets))
        print("linear layers affected:", len(selected))

        for rank in RANKS:
            lora_params = sum(
                rank * (in_features + out_features)
                for _, _, in_features, out_features in selected
            )

            percentage = (
                100 * lora_params / total_base_params
            )

            print(
                f"r={rank:2d}: "
                f"{lora_params:,} trainable params "
                f"({percentage:.4f}% of base)"
            )


if __name__ == "__main__":
    main()