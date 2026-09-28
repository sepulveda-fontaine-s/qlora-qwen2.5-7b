import math
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


import argparse


def collate_batch(examples, tokenizer):
    """Dynamically pad each batch to its longest sequence."""
    max_length = max(len(example["input_ids"]) for example in examples)

    input_ids = []
    attention_mask = []
    labels = []

    for example in examples:
        padding = max_length - len(example["input_ids"])

        input_ids.append(
            example["input_ids"]
            + [tokenizer.pad_token_id] * padding
        )

        attention_mask.append(
            example["attention_mask"]
            + [0] * padding
        )

        # Padding must not contribute to the loss.
        labels.append(
            example["labels"]
            + [-100] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def save_adapter(model, tokenizer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML training configuration.",
    )
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config["training"]["seed"]

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    model_path = config["model"]["path"]
    dataset_path = config["dataset"]["path"]

    output_dir = Path(config["output"]["dir"])
    checkpoint_dir = output_dir / "checkpoints"
    final_adapter_dir = output_dir / "final_adapter"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    print("=== LOAD DATASET ===")

    dataset = load_from_disk(dataset_path)

    train_dataset = dataset[
        config["dataset"]["train_split"]
    ]

    print("train examples:", len(train_dataset))

    batch_size = config["training"]["per_device_batch_size"]
    gradient_accumulation_steps = config["training"][
        "gradient_accumulation_steps"
    ]
    epochs = config["training"]["epochs"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_batch(
            examples,
            tokenizer,
        ),
        num_workers=2,
        pin_memory=True,
    )

    print("\n=== LOAD 4-BIT BASE MODEL ===")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=config["quantization"]["quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config["training"][
            "gradient_checkpointing"
        ],
    )

    print("\n=== ATTACH LORA ===")

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias=config["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=config["lora"]["target_modules"],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.train()

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("trainable parameters:", f"{trainable_params:,}")

    optimizer = AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    micro_batches_per_epoch = len(train_loader)

    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / gradient_accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        * config["training"]["warmup_ratio"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    print("\n=== TRAINING CONFIG ===")
    print("epochs:", epochs)
    print("batch size:", batch_size)
    print(
        "gradient accumulation:",
        gradient_accumulation_steps,
    )
    print(
        "effective batch size:",
        batch_size * gradient_accumulation_steps,
    )
    print(
        "optimizer steps:",
        total_optimizer_steps,
    )
    print("warmup steps:", warmup_steps)
    print(
        "learning rate:",
        config["training"]["learning_rate"],
    )

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    running_loss = 0.0
    accumulated_micro_batches = 0

    training_start = time.perf_counter()

    for epoch in range(epochs):

        print(f"\n=== EPOCH {epoch + 1}/{epochs} ===")

        for micro_step, batch in enumerate(
            train_loader,
            start=1,
        ):
            batch = {
                key: value.cuda(non_blocking=True)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss

            scaled_loss = (
                loss
                / gradient_accumulation_steps
            )

            scaled_loss.backward()

            running_loss += loss.item()
            accumulated_micro_batches += 1

            should_step = (
                micro_step % gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )

            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                config["training"]["max_grad_norm"],
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            average_loss = (
                running_loss
                / accumulated_micro_batches
            )

            if (
                global_step
                % config["training"]["logging_steps"]
                == 0
            ):
                elapsed = time.perf_counter() - training_start

                print(
                    f"step={global_step}/{total_optimizer_steps} "
                    f"loss={average_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8f} "
                    f"elapsed_min={elapsed / 60:.2f}"
                )

            running_loss = 0.0
            accumulated_micro_batches = 0

            if (
                global_step
                % config["training"]["checkpoint_steps"]
                == 0
            ):
                checkpoint_path = (
                    checkpoint_dir
                    / f"step_{global_step}"
                )

                print(
                    f"\nSaving checkpoint: "
                    f"{checkpoint_path}"
                )

                save_adapter(
                    model,
                    tokenizer,
                    checkpoint_path,
                )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    print("\n=== SAVE FINAL ADAPTER ===")

    save_adapter(
        model,
        tokenizer,
        final_adapter_dir,
    )

    print("\n=== TRAINING SUMMARY ===")
    print("optimizer steps:", global_step)
    print(
        "training time minutes:",
        round(training_seconds / 60, 2),
    )

    print(
        "peak GPU allocated GiB:",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            2,
        ),
    )

    print(
        "peak GPU reserved GiB:",
        round(
            torch.cuda.max_memory_reserved()
            / 1024**3,
            2,
        ),
    )

    print("final adapter:", final_adapter_dir)

    print("\n=== RESULT ===")
    print(
        f"QLoRA r={config['lora']['r']} training: SUCCESS"
    )


if __name__ == "__main__":
    main()