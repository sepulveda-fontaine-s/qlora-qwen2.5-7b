from prometheus_client import Counter, Histogram


REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total number of generation requests.",
    ["model_mode", "status"],
)

INFERENCE_LATENCY = Histogram(
    "llm_inference_latency_seconds",
    "Inference latency in seconds.",
    ["model_mode"],
)

GENERATED_TOKENS = Counter(
    "llm_generated_tokens_total",
    "Total number of generated tokens.",
    ["model_mode"],
)