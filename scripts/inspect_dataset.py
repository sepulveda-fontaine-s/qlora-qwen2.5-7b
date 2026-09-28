from collections import Counter

from datasets import load_dataset


DATASET_NAME = "databricks/databricks-dolly-15k"
CACHE_DIR = "data/raw/hf_cache"


def main() -> None:
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        cache_dir=CACHE_DIR,
    )

    print("\n=== DATASET SIZE ===")
    print(len(dataset))

    print("\n=== COLUMNS ===")
    print(dataset.column_names)

    print("\n=== FEATURES ===")
    print(dataset.features)

    print("\n=== CATEGORIES ===")
    categories = Counter(dataset["category"])
    for category, count in sorted(categories.items()):
        print(f"{category}: {count}")

    empty_contexts = sum(not context.strip() for context in dataset["context"])

    print("\n=== CONTEXT ===")
    print(f"Empty context: {empty_contexts}")
    print(f"Non-empty context: {len(dataset) - empty_contexts}")

    print("\n=== FIRST EXAMPLE ===")
    example = dataset[0]

    for key, value in example.items():
        print(f"\n[{key}]")
        print(value[:500] if isinstance(value, str) else value)


if __name__ == "__main__":
    main()