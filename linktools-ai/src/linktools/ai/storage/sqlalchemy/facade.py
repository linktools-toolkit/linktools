#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyStorage: the SQLAlchemy-backed Storage composition. Lives in its
own module so the core ``storage`` package (and ``linktools.ai`` itself) imports
cleanly without SQLAlchemy installed -- this module is only reached when a
caller actually requests ``SqlAlchemyStorage``. SQLAlchemy and
aiosqlite are optional dependencies; install via ``linktools-ai[sqlite]``.

All seven stores share one ``session_factory``; ``transaction()`` yields a
UnitOfWork whose stores bind to one AsyncSession + one transaction so a caller
can coordinate writes across stores atomically."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

try:  # optional dependency -- give a clear install hint instead of a raw ImportError
    from sqlalchemy.ext.asyncio import AsyncSession
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import-blocking test
    if exc.name and exc.name.split(".")[0] in {"sqlalchemy", "aiosqlite"}:
        raise ImportError(
            "SqlAlchemyStorage requires optional SQLAlchemy dependencies. "
            "Install with one of:\n"
            "  pip install 'linktools-ai[sqlite]'\n"
            "  pip install 'linktools-ai[sqlalchemy]'"
        ) from exc
    raise

from ...agent.approval import ApprovalStore
from ...events.store import EventStore
from ...memory.store import MemoryStore
from ...run.checkpoint import CheckpointStore
from ...run.store import RunStore
from ...session.store import SessionStore
from ...swarm.store import SwarmStore
from ...tool.idempotency import IdempotencyStore
from ..capabilities import SQLALCHEMY_STORAGE_CAPABILITIES
from ..facade import Storage
from ..resource.store import ResourceStore
from .approval import SqlAlchemyApprovalStore
from .checkpoint import SqlAlchemyCheckpointStore
from .event import SqlAlchemyEventStore
from .idempotency import SqlAlchemyIdempotencyStore
from .memory import SqlAlchemyMemoryStore
from .resource import SqlAlchemyResourceBackend
from .run import SqlAlchemyRunStore
from .session import SqlAlchemySessionStore
from .swarm import SqlAlchemySwarmStore


@dataclass(frozen=True)
class _UnitOfWork:
    """Atomic cross-store unit of work. Yielded by
    SqlAlchemyStorage.transaction(). All stores bind to the SAME AsyncSession,
    and that session's open transaction is owned by the surrounding
    ``async with`` -- writes through tx.runs / tx.approvals / etc. either all
    commit (clean exit) or all roll back (exception). Stores in UoW mode do NOT
    open their own sessions or call session.begin(); they reuse ``session`` and
    flush after each operation so subsequent reads within the unit observe prior
    writes."""

    session: AsyncSession
    runs: RunStore
    events: EventStore
    checkpoints: CheckpointStore
    approvals: ApprovalStore
    sessions: SessionStore
    swarms: SwarmStore
    memories: MemoryStore
    idempotency: IdempotencyStore


class SqlAlchemyStorage(Storage):
    """Storage backed by SQLAlchemy stores sharing one session_factory.
    Cross-cutting writes can be coordinated through transaction(), which yields
    a UnitOfWork whose stores share one AsyncSession + one transaction."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        resource_coordinator: "object | None" = None,
    ) -> None:
        super().__init__(
            resources=ResourceStore(primary=SqlAlchemyResourceBackend(session_factory=session_factory)),
            sessions=SqlAlchemySessionStore(session_factory=session_factory),
            runs=SqlAlchemyRunStore(session_factory=session_factory),
            events=SqlAlchemyEventStore(session_factory=session_factory),
            checkpoints=SqlAlchemyCheckpointStore(session_factory=session_factory),
            swarms=SqlAlchemySwarmStore(session_factory=session_factory),
            memories=SqlAlchemyMemoryStore(session_factory=session_factory),
            approvals=SqlAlchemyApprovalStore(session_factory=session_factory),
            idempotency=SqlAlchemyIdempotencyStore(session_factory=session_factory),
            capabilities=SQLALCHEMY_STORAGE_CAPABILITIES,
        )
        # Frozen dataclass: bypass __setattr__ to stash the factory for transaction().
        object.__setattr__(self, "_session_factory", session_factory)

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[_UnitOfWork]":
        """Yield a UnitOfWork whose stores all share one AsyncSession + one
        transaction. ``async with session.begin()`` auto-commits on clean exit
        and auto-rollbacks on exception, giving true atomicity across all
        stores: either every tx.* write persists, or none of them do."""
        async with self._session_factory() as session:
            async with session.begin():
                tx = _UnitOfWork(
                    session=session,
                    runs=SqlAlchemyRunStore(session_factory=self._session_factory, session=session),
                    events=SqlAlchemyEventStore(session_factory=self._session_factory, session=session),
                    checkpoints=SqlAlchemyCheckpointStore(session_factory=self._session_factory, session=session),
                    approvals=SqlAlchemyApprovalStore(session_factory=self._session_factory, session=session),
                    sessions=SqlAlchemySessionStore(session_factory=self._session_factory, session=session),
                    swarms=SqlAlchemySwarmStore(session_factory=self._session_factory, session=session),
                    memories=SqlAlchemyMemoryStore(session_factory=self._session_factory, session=session),
                    idempotency=SqlAlchemyIdempotencyStore(session_factory=self._session_factory, session=session),
                )
                yield tx
