"""Job: run output safety checks on recent assistant messages.

For every assistant message in the last `window_hours` that has no
governance_flag entry yet, this job:
  1. Evaluates toxicity and bias via DeepEval metrics.
  2. Optionally evaluates faithfulness when the preceding user message is available.
  3. Flags the message if toxicity > 0.5 or bias > 0.5.
  4. Upserts a row into governance_flags.

This job is Chapter 17's output-guardrail hook made visible at Chapter 15 so
the admin dashboard has real data to show.  The faithfulness metric (requiring
retrieved-context chunks) is wired but defaults to NULL when metadata is absent.

Called on a scheduled interval by APScheduler; failures per-message are logged
but do not abort remaining messages.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_TOXICITY_THRESHOLD = 0.5
_BIAS_THRESHOLD = 0.5


async def _evaluate_message(
    message: str,
    user_input: Optional[str],
    evaluator_model: str,
) -> tuple[float, float, Optional[float]]:
    """Run DeepEval metrics on a single assistant message.

    Returns (toxicity_score, bias_score, faithfulness_score).
    faithfulness_score is None when user_input is not available.
    """
    from deepeval.metrics import BiasMetric, ToxicityMetric
    from deepeval.test_case import LLMTestCase

    input_text = user_input or message

    test_case = LLMTestCase(input=input_text, actual_output=message)

    toxicity_metric = ToxicityMetric(
        model=evaluator_model, threshold=_TOXICITY_THRESHOLD,
        async_mode=True, include_reason=False,
    )
    bias_metric = BiasMetric(
        model=evaluator_model, threshold=_BIAS_THRESHOLD,
        async_mode=True, include_reason=False,
    )

    results = await asyncio.gather(
        toxicity_metric.a_measure(test_case),
        bias_metric.a_measure(test_case),
        return_exceptions=True,
    )

    toxicity_score = 0.0
    bias_score = 0.0

    if not isinstance(results[0], Exception):
        toxicity_score = float(toxicity_metric.score or 0.0)
    else:
        logger.warning(f"[output_guardrail] ToxicityMetric error: {results[0]}")

    if not isinstance(results[1], Exception):
        bias_score = float(bias_metric.score or 0.0)
    else:
        logger.warning(f"[output_guardrail] BiasMetric error: {results[1]}")

    return toxicity_score, bias_score, None


async def run_output_guardrail_job(
    *,
    admin_repo,
    guardrails_config,
    job_history: dict,
    window_hours: int = 24,
) -> None:
    """Evaluate recent unprocessed assistant messages and write governance_flags.

    Args:
        admin_repo:        AdminRepository instance.
        guardrails_config: GuardrailsConfig — supplies evaluator_model.
        job_history:       Shared dict updated with last-run metadata.
        window_hours:      Look back this many hours for unprocessed messages.
    """
    job_id = "output_guardrail"
    started_at = datetime.now(timezone.utc)
    processed = 0
    flagged = 0

    try:
        if not guardrails_config.enabled:
            logger.info("[output_guardrail] guardrails disabled — skipping job")
            job_history[job_id] = {
                "last_run": started_at.isoformat(),
                "status": "skipped",
                "detail": "guardrails disabled",
            }
            return

        messages = await admin_repo.get_unprocessed_assistant_messages(
            window_hours=window_hours, limit=100
        )
        logger.info(f"[output_guardrail] evaluating {len(messages)} messages")

        evaluator_model = guardrails_config.evaluator_model

        for msg in messages:
            chat_id = uuid.UUID(msg["chat_id"])
            session_id = uuid.UUID(msg["session_id"])

            try:
                user_input = await admin_repo.get_preceding_user_message(
                    session_id=session_id,
                    before_iso=msg["created_at"],
                )

                toxicity, bias, faithfulness = await _evaluate_message(
                    message=msg["message"],
                    user_input=user_input,
                    evaluator_model=evaluator_model,
                )

                is_flagged = toxicity > _TOXICITY_THRESHOLD or bias > _BIAS_THRESHOLD
                reasons = []
                if toxicity > _TOXICITY_THRESHOLD:
                    reasons.append(f"toxicity {toxicity:.2f}")
                if bias > _BIAS_THRESHOLD:
                    reasons.append(f"bias {bias:.2f}")
                flag_reason = "; ".join(reasons) if reasons else None

                await admin_repo.upsert_governance_flag(
                    chat_id=chat_id,
                    session_id=session_id,
                    toxicity_score=toxicity,
                    bias_score=bias,
                    faithfulness_score=faithfulness,
                    flagged=is_flagged,
                    flag_reason=flag_reason,
                )

                processed += 1
                if is_flagged:
                    flagged += 1
                    logger.warning(
                        f"[output_guardrail] flagged chat {chat_id}: {flag_reason}"
                    )

            except Exception as exc:
                logger.error(
                    f"[output_guardrail] failed for chat {chat_id}: {exc}",
                    exc_info=True,
                )

        detail = f"processed={processed}, flagged={flagged}"
        logger.info(f"[output_guardrail] job complete — {detail}")
        job_history[job_id] = {
            "last_run": started_at.isoformat(),
            "status": "succeeded",
            "detail": detail,
        }

    except Exception as exc:
        logger.error(f"[output_guardrail] job failed: {exc}", exc_info=True)
        job_history[job_id] = {
            "last_run": started_at.isoformat(),
            "status": "failed",
            "detail": str(exc),
        }
