import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
ADAPTER_PATH = "results/qlora_r8/final_adapter"
PROMPTS_PATH = (
    "results/generation_comparison/base_vs_r8_vs_r16.jsonl"
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:
        example = json.loads(next(file))

    inputs = tokenizer(
        example["prompt"],
        return_tensors="pt",
    ).to("cuda")

    print("=== LOAD BF16 + ADAPTER ===")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    peft_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    peft_model.eval()

    print("\n=== LOGITS BEFORE MERGE ===")

    with torch.inference_mode():
        before = peft_model(**inputs).logits[:, -1, :]

    before = before.float().cpu()

    before_token = before.argmax(dim=-1).item()

    print("argmax token id:", before_token)
    print(
        "argmax token:",
        repr(tokenizer.decode([before_token])),
    )

    print("\n=== MERGE ===")

    merged_model = peft_model.merge_and_unload(
        safe_merge=True
    )

    merged_model.eval()

    print("\n=== LOGITS AFTER MERGE ===")

    with torch.inference_mode():
        after = merged_model(**inputs).logits[:, -1, :]

    after = after.float().cpu()

    after_token = after.argmax(dim=-1).item()

    print("argmax token id:", after_token)
    print(
        "argmax token:",
        repr(tokenizer.decode([after_token])),
    )

    diff = (before - after).abs()

    print("\n=== NUMERICAL DIFFERENCE ===")
    print("max absolute logit diff:", diff.max().item())
    print("mean absolute logit diff:", diff.mean().item())
    print(
        "same next-token argmax:",
        before_token == after_token,
    )

    print("\n=== RESULT ===")
    print("Merge logit comparison: SUCCESS")


if __name__ == "__main__":
    main()