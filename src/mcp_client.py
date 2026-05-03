"""MCP tools library client.

Manages a persistent connection to the mcp-tools-library server and exposes
its tools as LangChain-compatible callables.

Lifecycle (managed by the API lifespan):
    loader = MCPToolLoader(config.mcp_config)
    await loader.connect()        # opens subprocess (stdio) or HTTP session
    ...
    await loader.disconnect()     # clean shutdown

Usage inside orchestrators:
    web_tools = loader.get_tools(["web_search", "fetch_webpage"])
    utility_tools = loader.get_tools(["calculate"])
    analysis_tools = loader.get_tools(["analyse"])
    extraction_tools = loader.get_tools(["rav_idp_process_and_ingest",
                                         "rav_idp_get_document_fidelity"])
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .core.models import MCPConfig

logger = logging.getLogger(__name__)


class MCPToolLoader:
    """Loads MCP tools from the tools library server and caches them in memory."""

    def __init__(self, config: MCPConfig) -> None:
        self._config = config
        self._tools: list[Any] = []
        self._connected = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Load all tools from the MCP server into memory.

        Uses the langchain-mcp-adapters 0.1.0+ API: tools are fetched via
        `await client.get_tools()` — no context manager required.
        """
        if not self._config.enabled:
            logger.info("MCP tools disabled (mcp.enabled=false). Skipping connection.")
            return

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            logger.error(
                "langchain-mcp-adapters is not installed. "
                "Add it to requirements.txt: %s", exc,
            )
            return

        server_cfg = self._build_server_config()
        client = MultiServerMCPClient(server_cfg)

        try:
            self._tools = await client.get_tools()
            self._connected = True
            tool_names = [t.name for t in self._tools]
            logger.info(
                "MCP tools loaded (%d tools): %s",
                len(self._tools),
                tool_names,
            )
        except Exception as exc:
            logger.error("Failed to connect to MCP tools server: %s", exc)
            self._connected = False

    async def disconnect(self) -> None:
        """No-op — tool lifecycle is managed internally by the MCP client."""
        self._connected = False
        logger.info("MCP tools server connection closed.")

    # ── Tool access ──────────────────────────────────────────────────────────

    def get_tools(self, names: list[str] | None = None) -> list[Any]:
        """Return loaded MCP tools, optionally filtered by name.

        Args:
            names: If provided, return only tools whose name is in this list.
                   If None, return all loaded tools.

        Returns:
            List of LangChain StructuredTool instances (empty if not connected).
        """
        if not self._connected or not self._tools:
            if names:
                logger.warning(
                    "MCP tools requested (%s) but server is not connected.", names
                )
            return []

        if names is None:
            return list(self._tools)

        tool_map = {t.name: t for t in self._tools}
        result = []
        for name in names:
            if name in tool_map:
                result.append(tool_map[name])
            else:
                logger.warning("MCP tool %r not found on server (available: %s)", name, list(tool_map))
        return result

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_server_config(self) -> dict:
        cfg = self._config

        if cfg.transport == "stdio":
            entry: dict = {
                "transport": "stdio",
                "command": cfg.stdio_command,
                "args": cfg.stdio_args,
            }
            # Merge explicitly configured env vars with the current process env
            # so the subprocess inherits PATH, HOME, etc.
            if cfg.stdio_env:
                merged_env = {**os.environ, **cfg.stdio_env}
                entry["env"] = merged_env
            return {"mcp-tools": entry}

        # streamable-http
        return {
            "mcp-tools": {
                "transport": "streamable_http",
                "url": cfg.http_url,
            }
        }
