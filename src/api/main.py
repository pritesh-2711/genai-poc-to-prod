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
from ..memory.repository import MemoryRepository
from ..databases.retrieval import PgVectorRetrievalRepository
from ..orchestrators import RAGOrchestrator
from ..reranker import CrossEncoderReranker
from .auth import router as auth_router
from .chat import router as chat_router
from .documents import router as documents_router
from .loader import BaseFileLoader, LocalFileLoader, S3FileLoader
from .sessions import router as sessions_router
from .state_backends import RedisClarificationStore
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
    st_cfg = config.storage_config

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

    # Reranker — loaded once at startup (model weights cached in memory)
    rr_cfg = config.reranker_config
    reranker = CrossEncoderReranker(model=rr_cfg.model, device=rr_cfg.device)

    # ── Storage + distributed state backends ─────────────────────────────────
    # local deployment  → local disk, in-process dict, MemorySaver checkpointer
    # cloud/aws         → S3, Redis dict wrapper, AsyncRedisSaver checkpointer

    file_loader: BaseFileLoader
    checkpointer = None  # None → RAGOrchestrator falls back to MemorySaver

    if st_cfg.deployment == "cloud" and st_cfg.cloud_provider == "aws":
        import redis
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        file_loader = S3FileLoader(
            bucket=st_cfg.aws_s3_bucket,
            region=st_cfg.aws_s3_region,
        )

        redis_client = redis.Redis.from_url(st_cfg.aws_redis_url, decode_responses=False)
        pending_clarifications = RedisClarificationStore(redis_client)

        checkpointer = AsyncRedisSaver.from_conn_string(st_cfg.aws_redis_url)
        await checkpointer.asetup()

        logger.info(
            f"Cloud storage: S3 bucket={st_cfg.aws_s3_bucket}, "
            f"Redis checkpointer + clarification store wired."
        )
    else:
        file_loader = LocalFileLoader()
        pending_clarifications: dict[str, str] = {}
        logger.info("Local storage: disk + in-process state backends.")

    # ── RAGOrchestrator ───────────────────────────────────────────────────────
    orchestrator = RAGOrchestrator(
        embedder=embedder,
        retrieval_repo=PgVectorRetrievalRepository(config.db_config),
        reranker=reranker,
        chat_service=chat_service,
        memory_repo=MemoryRepository(config.db_config),
        reranker_config=rr_cfg,
        chat_config=config.chat_config,
        checkpointer=checkpointer,  # None → falls back to MemorySaver inside orchestrator
    )

    app.state.config = config
    app.state.chat_service = chat_service
    app.state.embedder = embedder
    app.state.orchestrator = orchestrator
    app.state.file_loader = file_loader
    app.state.pending_clarifications = pending_clarifications

    logger.info(
        f"Application startup complete. "
        f"LLM={config.llm_config.provider}/{config.llm_config.model}, "
        f"Embedder={emb_cfg.provider}/{emb_cfg.model}, "
        f"Reranker={rr_cfg.model}, "
        f"Storage={st_cfg.deployment}"
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
app.include_router(documents_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}