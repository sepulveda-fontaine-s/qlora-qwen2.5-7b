from huggingface_hub import snapshot_download


MODEL_NAME = "Qwen/Qwen2.5-7B"
OUTPUT_DIR = "data/raw/models/Qwen2.5-7B"


def main() -> None:
    print(f"Downloading {MODEL_NAME}")
    print(f"Target directory: {OUTPUT_DIR}")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=OUTPUT_DIR,
    )

    print("\n=== DOWNLOAD COMPLETE ===")


if __name__ == "__main__":
    main()