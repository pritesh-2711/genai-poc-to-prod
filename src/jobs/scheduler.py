"""Factory for the APScheduler AsyncIOScheduler used by the FastAPI lifespan."""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..core.models import JobsConfig
from .chunk_scoring import run_chunk_scoring_job
from .intersession_memory import run_intersession_memory_job

logger = logging.getLogger(__name__)


def create_scheduler(
    *,
    jobs_config: JobsConfig,
    intersession_repo,
    chat_service,
    embedder,
) -> AsyncIOScheduler:
    """Build an AsyncIOScheduler pre-configured with all background jobs.

    Args:
        jobs_config:       JobsConfig parsed from config.yaml.
        intersession_repo: IntersessionRepository instance.
        chat_service:      ChatService used for LLM summarisation.
        embedder:          BaseEmbedder used to embed summaries.

    Returns:
        A configured (but not yet started) AsyncIOScheduler.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    if jobs_config.intersession.enabled:
        scheduler.add_job(
            run_intersession_memory_job,
            trigger="interval",
            hours=jobs_config.intersession.summary_interval_hours,
            kwargs={
                "intersession_repo": intersession_repo,
                "chat_service": chat_service,
                "embedder": embedder,
                "intersession_config": jobs_config.intersession,
            },
            id="intersession_memory",
            replace_existing=True,
            misfire_grace_time=3600,  # allow up to 1 h late start
        )
        logger.info(
            f"Intersession memory job scheduled every "
            f"{jobs_config.intersession.summary_interval_hours}h"
        )

    scheduler.add_job(
        run_chunk_scoring_job,
        trigger="interval",
        hours=jobs_config.chunk_scoring.interval_hours,
        kwargs={"intersession_repo": intersession_repo},
        id="chunk_scoring",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    logger.info(
        f"Chunk scoring job scheduled every {jobs_config.chunk_scoring.interval_hours}h"
    )

    return scheduler
