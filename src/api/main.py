"""FastAPI application factory."""

import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..chat_service import ChatService
from ..core.config import ConfigManager
from .auth import router as auth_router
from .chat import router as chat_router
from .sessions import router as sessions_router


def _setup_logging(logging_config_path: str = "configs/logging.yaml") -> None:
    config_path = Path(logging_config_path)
    if not config_path.exists():
        logging.basicConfig(level=logging.INFO)
        return
    with open(config_path) as f:
        config = yaml.safe_load(f)
    Path("logs").mkdir(exist_ok=True)
    logging.config.dictConfig(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise singletons once at startup, clean up on shutdown."""
    _setup_logging()
    logger = logging.getLogger(__name__)

    config = ConfigManager()
    chat_service = ChatService(
        llm_config=config.llm_config,
        chat_config=config.chat_config,
    )

    app.state.config = config
    app.state.chat_service = chat_service

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title="AI Research Assistant API",
    description="REST API for the GenAI research chat assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}