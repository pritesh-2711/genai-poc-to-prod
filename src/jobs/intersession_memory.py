"""Job: summarise every session's chat history and store embeddings for intersession memory."""

import logging
import uuid

from ..core.models import IntersessionConfig

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = (
    "You are a helpful assistant that creates concise conversation summaries."
)

_SUMMARY_PROMPT = """\
Summarise the following conversation between a user and an AI assistant in 3-5 sentences.
Focus on the main topics discussed, questions asked, and key conclusions reached.
Keep it factual and concise — this summary will provide background context in future sessions.

Conversation:
{history}

Summary:"""

# Rough character budget fed to the LLM (~4 chars per token, 4k token limit)
_MAX_HISTORY_CHARS = 16_000


async def run_intersession_memory_job(
    *,
    intersession_repo,
    chat_service,
    embedder,
    intersession_config: IntersessionConfig,
) -> None:
    """Summarise all sessions that have chat history and upsert summary + embedding.

    Called on a scheduled interval by APScheduler.  Failures per-session are
    logged but do not abort remaining sessions.
    """
    sessions = await intersession_repo.get_sessions_for_summary()
    logger.info(f"[intersession_memory] processing {len(sessions)} sessions")

    for entry in sessions:
        session_id = uuid.UUID(entry["session_id"])
        user_id = uuid.UUID(entry["user_id"])

        try:
            history_text = await intersession_repo.get_session_chat_history_text(session_id)
            if not history_text.strip():
                continue

            # Truncate from the end to fit within the LLM's context window
            if len(history_text) > _MAX_HISTORY_CHARS:
                history_text = history_text[-_MAX_HISTORY_CHARS:]

            prompt = _SUMMARY_PROMPT.format(history=history_text)
            summary = await chat_service.llm_provider.achat(
                user_message=prompt,
                system_prompt=_SUMMARY_SYSTEM,
            )
            summary = summary.strip()
            if not summary:
                continue

            token_count = len(summary) // 4  # rough approximation
            embedding = embedder.embed_one(summary)

            await intersession_repo.upsert_session_summary(
                user_id=user_id,
                session_id=session_id,
                summary_text=summary,
                embedding=embedding,
                token_count=token_count,
            )
            logger.debug(f"[intersession_memory] summarised session {session_id}")

        except Exception as exc:
            logger.error(
                f"[intersession_memory] failed for session {session_id}: {exc}",
                exc_info=True,
            )

    logger.info("[intersession_memory] job complete")
