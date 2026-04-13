"""Distributed state backends for multi-container deployments.

In a single-process (local) deployment, pending_clarifications is just a plain dict
and the LangGraph checkpointer is MemorySaver — both live in process RAM.

In a cloud (multi-container) deployment, both must be externalised:
  - RedisClarificationStore  — drop-in dict replacement backed by Redis.
  - AsyncRedisSaver          — LangGraph checkpointer backed by Redis (see main.py).

RedisClarificationStore is a synchronous, dict-compatible wrapper around redis.Redis.
Redis get/set/del calls are sub-millisecond on ElastiCache, so blocking is not a concern.
It implements the same interface used by chat.py:
    store[session_id] = thread_id
    thread_id = store.get(session_id)
    del store[session_id]
    session_id in store
    store.pop(session_id, default)
"""

import redis


class RedisClarificationStore:
    """Redis-backed replacement for app.state.pending_clarifications: dict[str, str].

    Keys are namespaced under "clarification:{session_id}" with a configurable TTL
    so interrupted threads don't accumulate indefinitely if a user never responds.

    Args:
        redis_client: A connected redis.Redis instance (sync).
        ttl:          Seconds before a pending clarification expires. Default 1 hour.
    """

    _PREFIX = "clarification:"

    def __init__(self, redis_client: redis.Redis, ttl: int = 3600) -> None:
        self._r = redis_client
        self._ttl = ttl

    # ------------------------------------------------------------------
    # dict-compatible interface (same operations used in chat.py)
    # ------------------------------------------------------------------

    def __setitem__(self, session_id: str, thread_id: str) -> None:
        self._r.setex(f"{self._PREFIX}{session_id}", self._ttl, thread_id)

    def __getitem__(self, session_id: str) -> str:
        val = self._r.get(f"{self._PREFIX}{session_id}")
        if val is None:
            raise KeyError(session_id)
        return val.decode() if isinstance(val, bytes) else val

    def __delitem__(self, session_id: str) -> None:
        self._r.delete(f"{self._PREFIX}{session_id}")

    def __contains__(self, session_id: str) -> bool:
        return self._r.exists(f"{self._PREFIX}{session_id}") > 0

    def get(self, session_id: str, default=None):
        try:
            return self[session_id]
        except KeyError:
            return default

    def pop(self, session_id: str, *args):
        val = self.get(session_id)
        if val is not None:
            del self[session_id]
            return val
        if args:
            return args[0]
        raise KeyError(session_id)
