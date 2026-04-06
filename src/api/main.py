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
from ..embedding import LocalEmbedder, OllamaEmbedder, OpenAIEmbedder
from ..guardrails import InputGuard
from .auth import router as auth_router
from .chat import router as chat_router
from .sessions import router as sessions_router
from .upload import router as upload_router


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

    input_guard = None
    if config.guardrails_config.enabled:
        input_guard = InputGuard(config.guardrails_config)

    chat_service = ChatService(
        llm_config=config.llm_config,
        chat_config=config.chat_config,
        input_guard=input_guard,
    )

    # Build embedder from config — loaded once, shared across all requests.
    emb_cfg = config.embedding_config
    _embedder_map = {
        "local":  lambda: LocalEmbedder(model=emb_cfg.model),
        "ollama": lambda: OllamaEmbedder(model=emb_cfg.model),
        "openai": lambda: OpenAIEmbedder(model=emb_cfg.model, api_key=emb_cfg.api_key),
    }
    embedder = _embedder_map[emb_cfg.provider]()

    app.state.config = config
    app.state.chat_service = chat_service
    app.state.embedder = embedder

    logger.info(
        f"Application startup complete. "
        f"LLM={config.llm_config.provider}/{config.llm_config.model}, "
        f"Embedder={emb_cfg.provider}/{emb_cfg.model}"
    )
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
app.include_router(upload_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}