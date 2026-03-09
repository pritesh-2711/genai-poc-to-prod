"""FastAPI application factory.

Mount all routers and configure CORS so the React dev server on
localhost:5173 (Vite) can communicate with the API on localhost:8000.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import router as auth_router
from .chat import router as chat_router
from .sessions import router as sessions_router

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Research Assistant API",
    description="REST API for the GenAI research chat assistant.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the Vite dev server and any deployed frontend origin.
# Tighten `allow_origins` when deploying to production.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(chat_router)


@app.get("/health", tags=["health"])
def health():
    """Quick liveness check."""
    return {"status": "ok"}
