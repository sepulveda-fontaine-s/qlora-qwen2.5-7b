from transformers import AutoConfig, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-7B"
CACHE_DIR = "data/raw/hf_cache"


def main() -> None:
    print(f"=== MODEL: {MODEL_NAME} ===")

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    print("\n=== ARCHITECTURE ===")
    print("model_type:", config.model_type)
    print("architectures:", config.architectures)
    print("hidden_size:", config.hidden_size)
    print("intermediate_size:", config.intermediate_size)
    print("num_hidden_layers:", config.num_hidden_layers)
    print("num_attention_heads:", config.num_attention_heads)
    print("num_key_value_heads:", config.num_key_value_heads)
    print("vocab_size:", config.vocab_size)
    print("max_position_embeddings:", config.max_position_embeddings)
    print("torch_dtype:", config.torch_dtype)

    print("\n=== TOKENIZER ===")
    print("tokenizer_class:", tokenizer.__class__.__name__)
    print("tokenizer_vocab_size:", tokenizer.vocab_size)
    print("bos_token:", tokenizer.bos_token)
    print("eos_token:", tokenizer.eos_token)
    print("pad_token:", tokenizer.pad_token)
    print("padding_side:", tokenizer.padding_side)

    print("\n=== CHAT TEMPLATE ===")
    print("has_chat_template:", tokenizer.chat_template is not None)


if __name__ == "__main__":
    main()