"""Chainlit UI with login, session management, and history-aware chat."""

import chainlit as cl

from src.chat_service import ChatService
from src.core.config import ConfigManager
from src.core.logging import LoggingManager
from src.memory import MemoryRepository
from src.memory.repository import AuthenticationError

logger = LoggingManager.setup()
config_manager = ConfigManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session_label(session) -> str:
    """Return a short display label for a session."""
    ts = session.created_at.strftime("%b %d, %H:%M")
    status = "" if session.is_active else " [ended]"
    return f"{session.session_name}{status} ({ts})"


async def _render_sessions_panel(repo: MemoryRepository, user_id, active_session_id) -> None:
    """Send an updated sessions panel as a single message with action buttons.

    Each session is presented as a clickable action. The active one is marked.
    Two extra actions are shown: 'New session' and 'End current session'.
    """
    sessions = repo.get_sessions(user_id)

    actions = []
    for s in sessions:
        label = _session_label(s)
        marker = " <<" if str(s.session_id) == str(active_session_id) else ""
        actions.append(
            cl.Action(
                name="switch_session",
                value=str(s.session_id),
                label=f"{label}{marker}",
            )
        )

    actions.append(cl.Action(name="new_session", value="new", label="+ New session"))
    actions.append(cl.Action(name="end_session", value="end", label="End current session"))

    await cl.Message(
        content="**Sessions** — click to switch:",
        actions=actions,
    ).send()


async def _load_and_display_history(repo: MemoryRepository, session_id) -> None:
    """Fetch history for a session and replay it into the chat UI."""
    history = repo.get_conversation_history(session_id)
    if not history:
        await cl.Message(content="_No messages in this session yet._").send()
        return

    for record in history:
        author = "You" if record.sender == "user" else "Assistant"
        await cl.Message(content=f"**{author}:** {record.message}").send()


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate the user against the database.

    Chainlit calls this with the credentials from its built-in login form.
    Return a cl.User on success or None on failure.
    """
    repo = MemoryRepository(config_manager.db_config)
    try:
        user_record = repo.authenticate_user(email=username, password=password)
        logger.info(f"Login successful for {username}")
        return cl.User(
            identifier=str(user_record.user_id),
            metadata={
                "name": user_record.name,
                "email": user_record.email,
            },
        )
    except AuthenticationError:
        logger.warning(f"Failed login attempt for {username}")
        return None


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Set up the session on first load after login."""
    cl_user = cl.user_session.get("user")
    user_id = cl_user.identifier
    user_name = cl_user.metadata.get("name", "there")

    repo = MemoryRepository(config_manager.db_config)
    chat_service = ChatService(
        llm_config=config_manager.llm_config,
        chat_config=config_manager.chat_config,
    )

    # Store service objects in the Chainlit user session
    cl.user_session.set("repo", repo)
    cl.user_session.set("chat_service", chat_service)
    cl.user_session.set("user_id", user_id)

    # Pick the most recent active session or create one
    sessions = repo.get_sessions(user_id)
    active_sessions = [s for s in sessions if s.is_active]

    if active_sessions:
        active_session = active_sessions[0]
    else:
        active_session = repo.create_session(user_id, "Session 1")

    cl.user_session.set("active_session_id", str(active_session.session_id))

    await cl.Message(
        content=(
            f"Welcome back, **{user_name}**!\n\n"
            f"Provider: `{config_manager.llm_config.provider}` | "
            f"Model: `{config_manager.llm_config.model}`\n\n"
            f"Active session: **{active_session.session_name}**"
        )
    ).send()

    await _render_sessions_panel(repo, user_id, active_session.session_id)

    # Replay existing history so the user sees prior messages
    await _load_and_display_history(repo, active_session.session_id)


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle a user message: persist, fetch history, get LLM response, persist."""
    repo: MemoryRepository = cl.user_session.get("repo")
    chat_service: ChatService = cl.user_session.get("chat_service")
    session_id = cl.user_session.get("active_session_id")

    if not repo or not chat_service or not session_id:
        await cl.Message(content="Session not initialised. Please refresh.").send()
        return

    user_message = message.content.strip()
    if not user_message:
        return

    # Persist user message
    repo.add_message(session_id, "user", user_message)

    # Fetch full conversation history to provide as context
    history = repo.get_conversation_history(session_id)
    # Exclude the message we just added — it will be passed as user_message
    context_history = history[:-1] if history else []

    # Get response from LLM with history context
    response = await chat_service.get_response_async(
        user_message=user_message,
        history=context_history,
    )

    # Persist assistant response
    repo.add_message(session_id, "assistant", response)

    await cl.Message(content=response).send()
    logger.info("Message processed and response sent.")


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

@cl.action_callback("switch_session")
async def on_switch_session(action: cl.Action):
    """Switch the active session."""
    repo: MemoryRepository = cl.user_session.get("repo")
    user_id = cl.user_session.get("user_id")
    new_session_id = action.value

    cl.user_session.set("active_session_id", new_session_id)

    sessions = repo.get_sessions(user_id)
    selected = next((s for s in sessions if str(s.session_id) == new_session_id), None)
    label = selected.session_name if selected else new_session_id

    await cl.Message(content=f"Switched to session: **{label}**").send()

    # Refresh the sessions panel to reflect the new active marker
    await _render_sessions_panel(repo, user_id, new_session_id)

    # Show history for the newly selected session
    await _load_and_display_history(repo, new_session_id)


@cl.action_callback("new_session")
async def on_new_session(action: cl.Action):
    """Create a new session and switch to it."""
    repo: MemoryRepository = cl.user_session.get("repo")
    user_id = cl.user_session.get("user_id")

    existing = repo.get_sessions(user_id)
    session_name = f"Session {len(existing) + 1}"
    new_session = repo.create_session(user_id, session_name)

    cl.user_session.set("active_session_id", str(new_session.session_id))

    await cl.Message(content=f"New session started: **{session_name}**").send()
    await _render_sessions_panel(repo, user_id, new_session.session_id)


@cl.action_callback("end_session")
async def on_end_session(action: cl.Action):
    """Mark the current session as inactive and start a new one."""
    repo: MemoryRepository = cl.user_session.get("repo")
    user_id = cl.user_session.get("user_id")
    session_id = cl.user_session.get("active_session_id")

    repo.terminate_session(session_id)

    existing = repo.get_sessions(user_id)
    session_name = f"Session {len(existing) + 1}"
    new_session = repo.create_session(user_id, session_name)

    cl.user_session.set("active_session_id", str(new_session.session_id))

    await cl.Message(
        content=f"Current session ended. Started new session: **{session_name}**"
    ).send()
    await _render_sessions_panel(repo, user_id, new_session.session_id)


@cl.on_chat_end
async def on_chat_end():
    """Clean up on disconnect."""
    cl.user_session.set("repo", None)
    cl.user_session.set("chat_service", None)
    logger.info("Chat session ended.")