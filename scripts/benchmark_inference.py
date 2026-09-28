import gc
import json
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


MODEL_PATH = "data/raw/models/Qwen2.5-7B"
ADAPTER_PATH = "results/qlora_r8/final_adapter"

PROMPTS_PATH = Path(
    "results/generation_comparison/base_vs_r8_vs_r16.jsonl"
)

OUTPUT_PATH = Path(
    "results/inference_benchmark/base_vs_qlora_r8.json"
)

MAX_NEW_TOKENS = 128


def percentile(values, p):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def load_model(adapter_path=None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

    model.eval()

    torch.cuda.synchronize()

    load_seconds = time.perf_counter() - start

    return model, load_seconds


def benchmark(label, adapter_path, prompts, tokenizer):
    print(f"\n=== {label.upper()} ===")

    model, load_seconds = load_model(adapter_path)

    print(f"model load seconds: {load_seconds:.3f}")

    # Warm-up: not included in benchmark timings.
    warmup_inputs = tokenizer(
        prompts[0],
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            min_new_tokens=MAX_NEW_TOKENS,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        
        )

    torch.cuda.synchronize()

    latencies = []
    generated_token_counts = []

    for number, prompt in enumerate(prompts, start=1):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        input_tokens = inputs["input_ids"].shape[1]

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

        generated_tokens = (
            output.shape[1] - input_tokens
        )

        latencies.append(elapsed)
        generated_token_counts.append(generated_tokens)

        print(
            f"{number}/{len(prompts)} "
            f"latency={elapsed:.3f}s "
            f"generated_tokens={generated_tokens}"
        )

    total_seconds = sum(latencies)
    total_tokens = sum(generated_token_counts)

    result = {
        "label": label,
        "model_load_seconds": load_seconds,
        "requests": len(prompts),
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "total_generated_tokens": total_tokens,
        "total_generation_seconds": total_seconds,
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "gpu_allocated_gib": (
            torch.cuda.memory_allocated() / 1024**3
        ),
        "gpu_reserved_gib": (
            torch.cuda.memory_reserved() / 1024**3
        ),
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
    }

    print("\n=== SUMMARY ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prompts = []

    with PROMPTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)
            prompts.append(record["prompt"])

    print("=== BENCHMARK SAMPLE ===")
    print("prompts:", len(prompts))

    results = {
        "base": benchmark(
            "base",
            None,
            prompts,
            tokenizer,
        ),
        "qlora_r8": benchmark(
            "qlora_r8",
            ADAPTER_PATH,
            prompts,
            tokenizer,
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n=== RESULT ===")
    print("Saved:", OUTPUT_PATH)
    print("Inference benchmark: SUCCESS")


if __name__ == "__main__":
    main()