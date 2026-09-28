import json
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
ADAPTER_PATH = "results/qlora_r8/final_adapter"
PROMPTS_PATH = (
    "results/generation_comparison/base_vs_r8_vs_r16.jsonl"
)

MAX_NEW_TOKENS = 64


def generate(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            min_new_tokens=MAX_NEW_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return text, elapsed


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:
        first_example = json.loads(next(file))

    prompt = first_example["prompt"]

    print("=== LOAD BF16 BASE ===")

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

    print("\n=== BEFORE MERGE ===")

    before_text, before_time = generate(
        peft_model,
        tokenizer,
        prompt,
    )

    print(f"latency: {before_time:.3f} s")
    print("output:")
    print(before_text)

    print("\n=== MERGE ===")

    merge_start = time.perf_counter()

    merged_model = peft_model.merge_and_unload(
        safe_merge=True
    )

    torch.cuda.synchronize()

    merge_time = time.perf_counter() - merge_start

    merged_model.eval()

    print(f"merge seconds: {merge_time:.3f}")

    print("\n=== AFTER MERGE ===")

    after_text, after_time = generate(
        merged_model,
        tokenizer,
        prompt,
    )

    print(f"latency: {after_time:.3f} s")
    print("output:")
    print(after_text)

    print("\n=== CHECK ===")

    print(
        "exact same generation:",
        before_text == after_text,
    )

    print(
        "allocated GPU GiB:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            3,
        ),
    )

    print(
        "peak GPU GiB:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3,
        ),
    )

    print("\n=== RESULT ===")
    print("BF16 merge smoke test: SUCCESS")


if __name__ == "__main__":
    main()