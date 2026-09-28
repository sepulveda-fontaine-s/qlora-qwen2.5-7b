import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "data/raw/models/Qwen2.5-7B"

PROMPTS_PATH = Path(
    "results/generation_comparison/base_vs_r8_vs_r16.jsonl"
)

OUTPUT_PATH = Path(
    "results/inference_benchmark/base_bf16.json"
)

MAX_NEW_TOKENS = 128


def percentile(values, p):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prompts = []

    with PROMPTS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            prompts.append(json.loads(line)["prompt"])

    print("=== LOAD BF16 BASE ===")

    load_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    print(f"load seconds: {load_seconds:.3f}")

    print("\n=== WARMUP ===")

    warmup_inputs = tokenizer(
        prompts[0],
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            min_new_tokens=16,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()

    latencies = []
    token_counts = []

    print("\n=== BENCHMARK ===")

    for number, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

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
        generated_tokens = output.shape[1] - input_length

        latencies.append(elapsed)
        token_counts.append(generated_tokens)

        print(
            f"{number}/{len(prompts)} "
            f"latency={elapsed:.3f}s "
            f"generated_tokens={generated_tokens}"
        )

    total_seconds = sum(latencies)
    total_tokens = sum(token_counts)

    result = {
        "label": "base_bf16",
        "load_seconds": load_seconds,
        "requests": len(prompts),
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "total_generated_tokens": total_tokens,
        "total_generation_seconds": total_seconds,
        "aggregate_tokens_per_second":
            total_tokens / total_seconds,
        "gpu_allocated_gib":
            torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gib":
            torch.cuda.memory_reserved() / 1024**3,
        "gpu_peak_allocated_gib":
            torch.cuda.max_memory_allocated() / 1024**3,
    }

    print("\n=== SUMMARY ===")

    for key, value in result.items():
        print(f"{key}: {value}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\n=== RESULT ===")
    print("Saved:", OUTPUT_PATH)
    print("Base BF16 benchmark: SUCCESS")


if __name__ == "__main__":
    main()