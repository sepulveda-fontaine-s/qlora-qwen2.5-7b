import logging

from fastapi import APIRouter, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import settings
from app.core.metrics import (
    GENERATED_TOKENS,
    INFERENCE_LATENCY,
    REQUEST_COUNT,
)
from app.schemas.generation import GenerationRequest, GenerationResponse


logger = logging.getLogger("app.inference")

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response):
    generation_graph = getattr(
        request.app.state,
        "generation_graph",
        None,
    )

    if generation_graph is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return {
            "status": "not_ready",
            "model_mode": settings.model_mode,
        }

    return {
        "status": "ready",
        "model_mode": settings.model_mode,
    }


@router.get("/model-info")
def model_info():
    return {
        "model_name": "Qwen/Qwen2.5-7B",
        "adapter": "QLoRA r8",
        "model_mode": settings.model_mode,
    }


@router.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.post(
    "/generate",
    response_model=GenerationResponse,
    responses={
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "internal_server_error",
                        "message": "An unexpected error occurred.",
                    }
                }
            },
        },
        503: {
            "description": "Model service not ready",
        },
    },
)
def generate(
    request_body: GenerationRequest,
    request: Request,
) -> GenerationResponse:
    generation_graph = getattr(
        request.app.state,
        "generation_graph",
        None,
    )

    if generation_graph is None:
        REQUEST_COUNT.labels(
            model_mode=settings.model_mode,
            status="error",
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not ready.",
        )

    try:
        result = generation_graph.invoke(
            {
                "request": request_body,
            }
        )

        response = result["response"]

    except Exception:
        REQUEST_COUNT.labels(
            model_mode=settings.model_mode,
            status="error",
        ).inc()

        raise

    REQUEST_COUNT.labels(
        model_mode=response.model_mode,
        status="success",
    ).inc()

    INFERENCE_LATENCY.labels(
        model_mode=response.model_mode,
    ).observe(response.latency_seconds)

    GENERATED_TOKENS.labels(
        model_mode=response.model_mode,
    ).inc(response.generated_tokens)

    logger.info(
        "inference_complete | model_mode=%s | latency_seconds=%.3f | generated_tokens=%d",
        response.model_mode,
        response.latency_seconds,
        response.generated_tokens,
    )

    return response