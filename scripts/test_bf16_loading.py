import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "data/raw/models/Qwen2.5-7B"


def gib(value: int) -> float:
    return value / 1024**3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()
   

    free_before, total = torch.cuda.mem_get_info(device)

    print("=== BEFORE LOAD ===")
    print("GPU:", torch.cuda.get_device_name(device))
    print(f"Total VRAM: {gib(total):.2f} GiB")
    print(f"Free VRAM: {gib(free_before):.2f} GiB")

    print("\n=== LOADING MODEL IN BF16 ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    free_after, _ = torch.cuda.mem_get_info(device)

    print("\n=== AFTER LOAD ===")
    print("Model dtype:", next(model.parameters()).dtype)
    print(f"Allocated: {gib(torch.cuda.memory_allocated(device)):.2f} GiB")
    print(f"Reserved: {gib(torch.cuda.memory_reserved(device)):.2f} GiB")
    print(f"Peak allocated: {gib(torch.cuda.max_memory_allocated(device)):.2f} GiB")
    print(f"Free VRAM: {gib(free_after):.2f} GiB")

    print("\n=== RESULT ===")
    print("BF16 model load: SUCCESS")


if __name__ == "__main__":
    main()