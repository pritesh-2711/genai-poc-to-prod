"""Job: recompute RLHF chunk quality scores from accumulated feedback counts."""

import logging

logger = logging.getLogger(__name__)


async def run_chunk_scoring_job(*, intersession_repo) -> None:
    """Recompute chunk_scores.score from positive/negative feedback counts.

    Uses Laplace smoothing: score = (pos + 1) / (pos + neg + 2).
    Called on a weekly schedule by APScheduler.
    """
    try:
        updated = await intersession_repo.recompute_chunk_scores()
        logger.info(f"[chunk_scoring] updated scores for {updated} chunks")
    except Exception as exc:
        logger.error(f"[chunk_scoring] job failed: {exc}", exc_info=True)
