import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n=== LOADING MODEL IN 4-BIT ===")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    free_after, _ = torch.cuda.mem_get_info(device)

    print("\n=== AFTER LOAD ===")
    print("Model quantized:", model.is_quantized)
    print(f"Allocated: {gib(torch.cuda.memory_allocated(device)):.2f} GiB")
    print(f"Reserved: {gib(torch.cuda.memory_reserved(device)):.2f} GiB")
    print(f"Peak allocated: {gib(torch.cuda.max_memory_allocated(device)):.2f} GiB")
    print(f"Free VRAM: {gib(free_after):.2f} GiB")

    print("\n=== QUANTIZATION CONFIG ===")
    print("4-bit:", quantization_config.load_in_4bit)
    print("quant type:", quantization_config.bnb_4bit_quant_type)
    print("compute dtype:", quantization_config.bnb_4bit_compute_dtype)
    print("double quant:", quantization_config.bnb_4bit_use_double_quant)

    print("\n=== RESULT ===")
    print("4-bit model load: SUCCESS")


if __name__ == "__main__":
    main()