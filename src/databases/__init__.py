"""Vector store repositories — asyncpg-backed ingestion and retrieval.

Two ABCs define the interface, two concrete classes implement it:

    BaseIngestionRepository      → PgVectorIngestionRepository
    BaseRetrievalRepository      → PgVectorRetrievalRepository

Both are initialized with a DBConfig and expose async methods only.

Example:
    from src.databases import PgVectorIngestionRepository, PgVectorRetrievalRepository
    from src.core.config import ConfigManager

    cfg = ConfigManager()
    ingest_repo   = PgVectorIngestionRepository(cfg.db_config)
    retrieve_repo = PgVectorRetrievalRepository(cfg.db_config)

    parent_uuids, child_uuids = await ingest_repo.ingest_documents(...)
    results  = await retrieve_repo.search(query_embedding, top_k=5)
    contexts = await retrieve_repo.fetch_parent_contexts(
        [r["parent_id"] for r in results if r["parent_id"]]
    )
"""

from .base import BaseIngestionRepository, BaseRetrievalRepository
from .ingestion import IngestionRepositoryError, PgVectorIngestionRepository
from .pipeline import IngestionPipeline, IngestionPipelineError, IngestionResult
from .retrieval import RetrievalRepositoryError, PgVectorRetrievalRepository

__all__ = [
    "BaseIngestionRepository",
    "BaseRetrievalRepository",
    "PgVectorIngestionRepository",
    "PgVectorRetrievalRepository",
    "IngestionRepositoryError",
    "RetrievalRepositoryError",
    "IngestionPipeline",
    "IngestionPipelineError",
    "IngestionResult",
]
