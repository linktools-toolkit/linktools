#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owner-fixed Runtime mutation coordinators for concrete adapters."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from linktools.core import environ

from .._domain import RuntimeDomain

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_logger = environ.get_logger("ai.runtime.state.transaction")


Callback = Callable[[], Awaitable[None] | None]


class TransactionHub:
    """Shared task ownership state for the domain-specific coordinators."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._domain: RuntimeDomain | None = None
        self._depth = 0
        self._dirty = False
        self._snapshot: Callable[[], object] | None = None
        self._restore: Callable[[object], None] | None = None
        self._commit: Callable[[RuntimeDomain], Awaitable[None] | None] | None = None
        self._rollback: Callable[[frozenset[RuntimeDomain]], Awaitable[None] | None] | None = None
        self._snapshot_value: object | None = None

    def configure(
        self,
        *,
        snapshot: Callable[[], object] | None = None,
        restore: Callable[[object], None] | None = None,
        commit: Callable[[RuntimeDomain], Awaitable[None] | None] | None = None,
        rollback: Callable[[frozenset[RuntimeDomain]], Awaitable[None] | None] | None = None,
    ) -> None:
        self._snapshot = snapshot
        self._restore = restore
        self._commit = commit
        self._rollback = rollback

    @property
    def configured(self) -> bool:
        return self._snapshot is not None or self._restore is not None or self._commit is not None or self._rollback is not None

    @property
    def active_task(self) -> asyncio.Task[object] | None:
        return self._owner

    @property
    def active_domain(self) -> RuntimeDomain | None:
        return self._domain

    @property
    def depth(self) -> int:
        return self._depth

    async def enter(self, domain: RuntimeDomain) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("runtime transaction requires an asyncio task")
        if task is self._owner:
            if self._domain is not domain:
                _logger.error(
                    "cross-domain nested mutation rejected: active=%s attempted=%s",
                    self._domain.value if self._domain is not None else "none",
                    domain.value,
                )
                raise RuntimeError("cross-domain runtime mutation")
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._domain = domain
        self._depth = 1
        self._dirty = False
        self._snapshot_value = None if self._snapshot is None else self._snapshot()

    async def exit(self, domain: RuntimeDomain, exc_type: object) -> None:
        if asyncio.current_task() is not self._owner or self._domain is not domain:
            raise RuntimeError("runtime transaction owner mismatch")
        self._depth -= 1
        if self._depth != 0:
            return
        dirty = self._dirty
        snapshot = self._snapshot_value
        self._dirty = False
        self._snapshot_value = None
        try:
            if exc_type is not None:
                await self._rollback_transaction(frozenset({domain}) if dirty else frozenset(), snapshot)
            elif dirty:
                try:
                    if self._commit is not None:
                        await _invoke(self._commit, domain)
                except BaseException:
                    await self._rollback_transaction(frozenset({domain}), snapshot)
                    raise
        finally:
            self._owner = None
            self._domain = None
            self._lock.release()

    async def _rollback_transaction(self, domains: frozenset[RuntimeDomain], snapshot: object | None) -> None:
        if self._rollback is not None:
            await _invoke(self._rollback, domains)
        if snapshot is not None and self._restore is not None:
            self._restore(snapshot)

    def mark_changed(self, domain: RuntimeDomain) -> None:
        task = asyncio.current_task()
        if task is None or task is not self._owner or self._depth <= 0 or self._domain is not domain:
            _logger.error("Runtime mutation marked outside owner transaction: domain=%s", domain.value)
            raise RuntimeError("storage mutation outside transaction")
        self._dirty = True

    @property
    def snapshot_value(self) -> object | None:
        return self._snapshot_value


async def _invoke(callback: Callable[..., Awaitable[None] | None], *args: object) -> None:
    value = callback(*args)
    if inspect.isawaitable(value):
        await value


class RuntimeTransactionCoordinator:
    def __init__(self, owner_domain: RuntimeDomain, *, hub: TransactionHub) -> None:
        if not isinstance(owner_domain, RuntimeDomain):
            raise ValueError("transaction owner domain is invalid")
        self._owner_domain = owner_domain
        self._hub = hub

    @property
    def owner_domain(self) -> RuntimeDomain:
        return self._owner_domain

    @property
    def hub(self) -> TransactionHub:
        return self._hub

    @asynccontextmanager
    async def mutation(self) -> "AsyncIterator[None]":
        await self._hub.enter(self._owner_domain)
        try:
            yield
        except BaseException as error:
            await self._hub.exit(self._owner_domain, type(error))
            raise
        else:
            await self._hub.exit(self._owner_domain, None)

    async def __aenter__(self) -> "RuntimeTransactionCoordinator":
        await self._hub.enter(self._owner_domain)
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self._hub.exit(self._owner_domain, exc_type)

    def mark_changed(self) -> None:
        self._hub.mark_changed(self._owner_domain)


__all__ = ["RuntimeTransactionCoordinator", "TransactionHub"]
