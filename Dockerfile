FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_MODE=mock \
    MODEL_PATH=/models/Qwen2.5-7B \
    ADAPTER_PATH=/adapters/qlora_r8

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install torch \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python -m pip install .

COPY app/ ./app/

EXPOSE 8040

CMD [
    "python",
    "-m",
    "uvicorn",
    "app.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8040"
]