"""Factory for the APScheduler AsyncIOScheduler used by the FastAPI lifespan."""

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from ..core.models import GuardrailsConfig, JobsConfig
from .chunk_scoring import run_chunk_scoring_job
from .intersession_memory import run_intersession_memory_job
from .output_guardrail import run_output_guardrail_job

logger = logging.getLogger(__name__)


def _make_tracked(job_func, job_id: str, job_history: dict):
    """Wrap a job coroutine so it records last-run / status in job_history."""

    async def _wrapper(**kwargs):
        started = datetime.now(timezone.utc)
        try:
            await job_func(**kwargs)
            if job_id not in job_history:
                # job updated job_history itself (output_guardrail does this)
                job_history[job_id] = {
                    "last_run": started.isoformat(),
                    "status": "succeeded",
                    "detail": "",
                }
        except Exception as exc:
            job_history[job_id] = {
                "last_run": started.isoformat(),
                "status": "failed",
                "detail": str(exc),
            }
            raise

    return _wrapper


def create_scheduler(
    *,
    jobs_config: JobsConfig,
    guardrails_config: GuardrailsConfig,
    intersession_repo,
    admin_repo,
    chat_service,
    embedder,
    job_history: dict,
) -> AsyncIOScheduler:
    """Build an AsyncIOScheduler pre-configured with all background jobs.

    Args:
        jobs_config:        JobsConfig parsed from config.yaml.
        guardrails_config:  GuardrailsConfig — passed to output_guardrail job.
        intersession_repo:  IntersessionRepository instance.
        admin_repo:         AdminRepository instance.
        chat_service:       ChatService used for LLM summarisation.
        embedder:           BaseEmbedder used to embed summaries.
        job_history:        Shared dict (app.state.job_history) updated per run.

    Returns:
        A configured (but not yet started) AsyncIOScheduler.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    # ── Intersession memory ───────────────────────────────────────────────────
    if jobs_config.intersession.enabled:
        scheduler.add_job(
            _make_tracked(run_intersession_memory_job, "intersession_memory", job_history),
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
            misfire_grace_time=3600,
        )
        logger.info(
            f"Intersession memory job scheduled every "
            f"{jobs_config.intersession.summary_interval_hours}h"
        )

    # ── Chunk scoring (RLHF) ─────────────────────────────────────────────────
    scheduler.add_job(
        _make_tracked(run_chunk_scoring_job, "chunk_scoring", job_history),
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

    # ── Output guardrail ──────────────────────────────────────────────────────
    scheduler.add_job(
        run_output_guardrail_job,
        trigger="interval",
        hours=1,
        kwargs={
            "admin_repo": admin_repo,
            "guardrails_config": guardrails_config,
            "job_history": job_history,
        },
        id="output_guardrail",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    logger.info("Output guardrail job scheduled every 1h")

    return scheduler
