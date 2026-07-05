#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage facade (spec docs/linktools-ai.md section 11): composes the Phase-1/2
resource/session/run/event/checkpoint backends into one frozen dataclass so a
caller gets a single object that can do everything. Two constructors cover the
two supported deployment shapes:

- FileStorage: five independent file backends under a root dir. No cross-store
  transactions are possible (each backend owns its own files), so the inherited
  Storage.transaction() raises StorageCapabilityError.
- SqlAlchemyStorage: five sqlalchemy backends sharing one session_factory, plus
  an overridden transaction() that yields a shared AsyncSession so a caller can
  coordinate writes across stores in one DB transaction.

Subclasses use object.__setattr__ to stash their own state (e.g. the session
factory) because the dataclass is frozen -- hence frozen=True rather than
slots=True, which would also forbid per-subclass attributes."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from ..errors import StorageCapabilityError
from ..events.store import EventStore
from ..run.checkpoint import CheckpointStore
from ..run.store import RunStore
from ..session.store import SessionStore
from ..swarm_runtime.store import SwarmStore
from .capabilities import FILE_STORAGE_CAPABILITIES, SQLALCHEMY_STORAGE_CAPABILITIES, StorageCapabilities
from .file.checkpoint import FileCheckpointStore
from .file.event import FileEventStore
from .file.run import FileRunStore
from .file.session import FileSessionStore
from .file.swarm import FileSwarmStore
from .resource.file import FileResourceBackend
from .resource.store import ResourceStore
from .sqlalchemy.checkpoint import SqlAlchemyCheckpointStore
from .sqlalchemy.event import SqlAlchemyEventStore
from .sqlalchemy.resource import SqlAlchemyResourceBackend
from .sqlalchemy.run import SqlAlchemyRunStore
from .sqlalchemy.session import SqlAlchemySessionStore
from .sqlalchemy.swarm import SqlAlchemySwarmStore


@dataclass(frozen=True)
class Storage:
    """Frozen composition of the five storage backends. Concrete subclasses
    (FileStorage, SqlAlchemyStorage) are responsible for constructing the
    backends; this base only holds them and exposes the cross-cutting
    transaction() hook."""

    resources: ResourceStore
    sessions: SessionStore
    runs: RunStore
    events: EventStore
    checkpoints: CheckpointStore
    swarms: SwarmStore
    capabilities: StorageCapabilities

    def transaction(self) -> "AsyncIterator[AsyncSession]":
        """Cross-store transactional scope. The base implementation always
        raises: only a Storage whose backends genuinely share one underlying
        transaction provider (e.g. SqlAlchemyStorage) can honor this. Callers
        should branch on capabilities.cross_store_transactions before relying
        on it.

        Intentionally a plain ``def`` (not ``async def``): raising here instead
        of inside ``__aenter__`` means ``async with storage.transaction()``
        fails at the call site with StorageCapabilityError, not with a
        confusing ``TypeError`` about a coroutine not supporting the async
        context-manager protocol."""
        raise StorageCapabilityError(
            f"{type(self).__name__} does not support cross-store transactions"
        )


class FileStorage(Storage):
    """Storage backed by five independent file-system backends. Each backend
    manages its own files, so cross-store transactions are NOT available --
    transaction() raises StorageCapabilityError. Branch on
    capabilities.cross_store_transactions (False here) before calling it."""

    def __init__(self, *, root: "str | Path" = "./data") -> None:
        root_path = Path(root)
        super().__init__(
            resources=ResourceStore(primary=FileResourceBackend(root=root_path / "resources")),
            sessions=FileSessionStore(root=root_path / "sessions"),
            runs=FileRunStore(root=root_path / "runs"),
            events=FileEventStore(root=root_path / "events"),
            checkpoints=FileCheckpointStore(root=root_path / "checkpoints"),
            swarms=FileSwarmStore(root=root_path / "swarms"),
            capabilities=FILE_STORAGE_CAPABILITIES,
        )


class SqlAlchemyStorage(Storage):
    """Storage backed by five sqlalchemy backends sharing one session_factory.
    All cross-cutting writes can be coordinated through transaction(), which
    yields a single AsyncSession the caller can use directly. The
    resource_coordinator parameter is accepted for forward compatibility with
    future cross-store coordination features but is not wired in this phase."""

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
            capabilities=SQLALCHEMY_STORAGE_CAPABILITIES,
        )
        # Frozen dataclass: bypass __setattr__ to stash the factory for transaction().
        object.__setattr__(self, "_session_factory", session_factory)

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[AsyncSession]":
        async with self._session_factory() as session:
            yield session
