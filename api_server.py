"""Uvicorn entry point for the FastAPI REST API.

Run from the project root:

    python api_server.py

or directly with uvicorn:

    uvicorn src.api.main:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["*.log"],
    )
