from datasets import load_dataset, DatasetDict


DATASET_NAME = "databricks/databricks-dolly-15k"
CACHE_DIR = "data/raw/hf_cache"
OUTPUT_DIR = "data/processed/dolly_15k"
SEED = 42
VALIDATION_SIZE = 0.10


def main() -> None:
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=CACHE_DIR,
    )

    # Convert the string category column into a ClassLabel column.
    # This allows Hugging Face Datasets to perform stratified splitting.
    dataset = dataset.class_encode_column("category")

    split = dataset.train_test_split(
        test_size=VALIDATION_SIZE,
        seed=SEED,
        stratify_by_column="category",
    )

    prepared = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )

    prepared.save_to_disk(OUTPUT_DIR)

    print("=== SPLIT SIZES ===")
    print("train:", len(prepared["train"]))
    print("validation:", len(prepared["validation"]))

    print("\n=== CATEGORY LABELS ===")
    print(prepared["train"].features["category"].names)

    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()