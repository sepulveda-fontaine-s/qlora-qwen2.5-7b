from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.exceptions import global_exception_handler
from app.core.logging_config import configure_logging
from app.services.factory import get_model_service
from app.workflows.generation_graph import build_generation_graph


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = get_model_service()
    generation_graph = build_generation_graph(model_service)

    app.state.model_service = model_service
    app.state.generation_graph = generation_graph

    yield

    app.state.generation_graph = None
    app.state.model_service = None


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8050",
        "http://localhost:8050",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)
app.include_router(router)