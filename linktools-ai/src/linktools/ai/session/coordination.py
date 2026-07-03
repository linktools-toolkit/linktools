#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import uuid
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

# `RedisSessionLease`/`RedisSessionCoordinator`/`coordinator_for_redis` below are
# structurally typed against any client object exposing the try_acquire/get/setex/
# delete/release_if_owner surface — no concrete Redis client type is imported here,
# so this module has no runtime or type-only dependency on a Redis package.
RedisClient = Any

_MAX_PERSIST_MEMORY_PER_SESSION = 256


class SessionLeaseConflictError(RuntimeError):
    """Raised when a second live lease is requested for the same session."""


class SessionLeaseHandle(Protocol):
    async def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PersistDecision:
    should_persist: bool
    history_already_committed: bool = False


class SessionCoordinator(Protocol):
    async def acquire_lease(self, session_id: str) -> SessionLeaseHandle: ...

    async def begin_persist(self, session_id: str, idempotency_key: "str | None") -> PersistDecision: ...

    async def complete_persist(
        self,
        session_id: str,
        idempotency_key: "str | None",
        *,
        history_committed: bool,
        completed: bool,
    ) -> None: ...


@dataclass(slots=True)
class InMemorySessionLease:
    _coordinator: "InMemorySessionCoordinator"
    session_id: str
    lease_id: str
    _released: bool = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._coordinator._release(self.session_id, self.lease_id)


class InMemorySessionCoordinator:
    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._leases: "dict[str, str]" = {}
        self._inflight_persists: "set[tuple[str, str]]" = set()
        self._history_committed_persists: "dict[str, OrderedDict[str, None]]" = {}
        self._completed_persists: "dict[str, OrderedDict[str, None]]" = {}

    async def acquire_lease(self, session_id: str) -> InMemorySessionLease:
        async with self._guard:
            if session_id in self._leases:
                raise SessionLeaseConflictError(f"session lease already held: {session_id}")
            lease_id = uuid.uuid4().hex
            self._leases[session_id] = lease_id
            return InMemorySessionLease(self, session_id, lease_id)

    async def begin_persist(self, session_id: str, idempotency_key: "str | None") -> PersistDecision:
        if not idempotency_key:
            return PersistDecision(should_persist=True)
        persist_key = (session_id, idempotency_key)
        async with self._guard:
            completed = self._completed_persists.get(session_id)
            history_committed = self._history_committed_persists.get(session_id)
            if (completed is not None and idempotency_key in completed) or persist_key in self._inflight_persists:
                return PersistDecision(should_persist=False)
            self._inflight_persists.add(persist_key)
            return PersistDecision(
                should_persist=True,
                history_already_committed=history_committed is not None and idempotency_key in history_committed,
            )

    async def complete_persist(
        self,
        session_id: str,
        idempotency_key: "str | None",
        *,
        history_committed: bool,
        completed: bool,
    ) -> None:
        if not idempotency_key:
            return
        persist_key = (session_id, idempotency_key)
        async with self._guard:
            self._inflight_persists.discard(persist_key)
            if history_committed:
                self._remember(self._history_committed_persists, session_id, idempotency_key)
            if completed:
                self._remember(self._completed_persists, session_id, idempotency_key)

    async def _release(self, session_id: str, lease_id: str) -> None:
        async with self._guard:
            if self._leases.get(session_id) == lease_id:
                self._leases.pop(session_id, None)

    @staticmethod
    def _remember(bucket: "dict[str, OrderedDict[str, None]]", session_id: str, idempotency_key: str) -> None:
        session_keys = bucket.setdefault(session_id, OrderedDict())
        session_keys.pop(idempotency_key, None)
        session_keys[idempotency_key] = None
        while len(session_keys) > _MAX_PERSIST_MEMORY_PER_SESSION:
            session_keys.popitem(last=False)

@dataclass(slots=True)
class RedisSessionLease:
    _redis: "RedisClient"
    lock_key: str
    lock_val: str
    _released: bool = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._redis.release_if_owner(self.lock_key, self.lock_val)


class RedisSessionCoordinator:
    """Cross-process session coordination backed by Redis.

    `InMemorySessionCoordinator` only serializes persists/leases within one pod's
    process memory — two pods handling the same `session_id` (the normal case for a
    DB-backed chat session, which is built to "continue on any pod") never see each
    other's in-flight state, so both can pass `begin_persist`/`acquire_lease` at once
    and race writing `agent_session_events`. This coordinator makes the same decisions
    visible across pods via Redis `SET NX EX`, the same primitive `cap_write_lock`
    (engine/infra/capability.py) already uses for cross-pod capability-file locks.
    """

    _LEASE_TTL_SECONDS = 30  # bounds one persist+sidecar-write cycle, not an LLM call
    _PERSIST_DONE_TTL_SECONDS = 24 * 3600

    def __init__(self, redis: "RedisClient") -> None:
        self._redis = redis

    async def acquire_lease(self, session_id: str) -> RedisSessionLease:
        lock_key = f"session:lease:{session_id}"
        lock_val = uuid.uuid4().hex
        if not await self._redis.try_acquire(lock_key, lock_val, self._LEASE_TTL_SECONDS):
            raise SessionLeaseConflictError(f"session lease already held: {session_id}")
        return RedisSessionLease(self._redis, lock_key, lock_val)

    async def begin_persist(self, session_id: str, idempotency_key: "str | None") -> PersistDecision:
        if not idempotency_key:
            return PersistDecision(should_persist=True)
        done_key = self._done_key(session_id, idempotency_key)
        if await self._redis.get(done_key) is not None:
            return PersistDecision(should_persist=False, history_already_committed=True)
        inflight_key = self._inflight_key(session_id, idempotency_key)
        if not await self._redis.try_acquire(inflight_key, "1", self._LEASE_TTL_SECONDS):
            return PersistDecision(should_persist=False)
        history_committed = await self._redis.get(self._history_key(session_id, idempotency_key)) is not None
        return PersistDecision(should_persist=True, history_already_committed=history_committed)

    async def complete_persist(
        self,
        session_id: str,
        idempotency_key: "str | None",
        *,
        history_committed: bool,
        completed: bool,
    ) -> None:
        if not idempotency_key:
            return
        await self._redis.delete(self._inflight_key(session_id, idempotency_key))
        if history_committed:
            await self._redis.setex(self._history_key(session_id, idempotency_key), self._PERSIST_DONE_TTL_SECONDS, "1")
        if completed:
            await self._redis.setex(self._done_key(session_id, idempotency_key), self._PERSIST_DONE_TTL_SECONDS, "1")

    @staticmethod
    def _inflight_key(session_id: str, idempotency_key: str) -> str:
        return f"session:persist:inflight:{session_id}:{idempotency_key}"

    @staticmethod
    def _history_key(session_id: str, idempotency_key: str) -> str:
        return f"session:persist:history:{session_id}:{idempotency_key}"

    @staticmethod
    def _done_key(session_id: str, idempotency_key: str) -> str:
        return f"session:persist:done:{session_id}:{idempotency_key}"


def coordinator_for_redis(redis: "RedisClient | None", store: object) -> "SessionCoordinator":
    """Pick the right coordination backend for a DB-backed (cross-pod) session.

    Redis enabled -> cross-pod safe `RedisSessionCoordinator`. Redis disabled/absent ->
    fall back to the store-keyed in-memory coordinator (single-instance dev only,
    matching the fallback convention `cap_write_lock` already uses for capability-file
    locks) so same-pod calls for the same store still share lease/persist state.
    """
    if redis is not None and getattr(redis.config, "enabled", False):
        return RedisSessionCoordinator(redis)
    return coordinator_for_store(store)


_STORE_COORDINATORS: "weakref.WeakKeyDictionary[object, InMemorySessionCoordinator]" = weakref.WeakKeyDictionary()


def _coordinator_store_key(store: object) -> object:
    backend = getattr(store, "_db", None)
    if backend is None:
        return store
    try:
        weakref.ref(backend)
    except TypeError:
        return store
    return backend


def coordinator_for_store(store: object) -> InMemorySessionCoordinator:
    key = _coordinator_store_key(store)
    coordinator = _STORE_COORDINATORS.get(key)
    if coordinator is None:
        coordinator = InMemorySessionCoordinator()
        _STORE_COORDINATORS[key] = coordinator
    return coordinator
