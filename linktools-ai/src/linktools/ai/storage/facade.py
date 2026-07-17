#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage facade: composes the storage
backends into one frozen dataclass so a caller gets a single object that can do
everything.

This module is deliberately SQLAlchemy-free: ``Storage`` and
``FileStorage`` depend only on the standard library and core stores, so
``import linktools.ai`` and ``import linktools.ai.storage`` succeed without the
optional SQLAlchemy/aiosqlite dependencies. The SQLAlchemy-backed composition
(``SqlAlchemyStorage``) lives in ``linktools.ai.storage.sqlalchemy`` and is
loaded lazily via ``storage/__init__.__getattr__``.

- Storage: frozen composition of the nine backends + capabilities; the base
  ``transaction()`` raises StorageCapabilityError (only a Storage whose backends
  genuinely share one transaction provider -- e.g. SqlAlchemyStorage -- honors
  it).
- FileStorage: nine independent file backends under a root dir. No cross-store
  transactions are possible, so the inherited transaction() raises.

Subclasses use object.__setattr__ to stash their own state (e.g. the session
factory) because the dataclass is frozen -- hence frozen=True rather than
slots=True, which would also forbid per-subclass attributes."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ...evaluation.store import EvalStore
    from ..task.store import TaskStore
    from .sqlalchemy.facade import _UnitOfWork

from ..agent.approval import ApprovalStore
from ..errors import StorageCapabilityError
from ..events.store import EventStore
from ..memory.store import MemoryStore
from ..run.checkpoint import CheckpointStore
from ..run.definition import RunDefinitionStore
from ..run.store import RunStore
from ..session.store import SessionStore
from ..swarm.store import SwarmStore
from ..tool.idempotency import IdempotencyStore
from .capabilities import FILE_STORAGE_CAPABILITIES, StorageCapabilities
from .file.approval import FileApprovalStore
from .file.checkpoint import FileCheckpointStore
from .file.definition import FileRunDefinitionStore
from .file.event import FileEventStore
from .file.idempotency import FileIdempotencyStore
from .file.memory import FileMemoryStore
from .file.run import FileRunStore
from .file.session import FileSessionStore
from .file.swarm import FileSwarmStore
from .resource.file import FileResourceBackend
from .resource.store import ResourceStore


@dataclass(frozen=True)
class Storage:
    """Frozen composition of the storage backends. Concrete subclasses
    (FileStorage, SqlAlchemyStorage) are responsible for constructing the
    backends; this base only holds them and exposes the cross-cutting
    transaction() hook."""

    resources: ResourceStore
    sessions: SessionStore
    runs: RunStore
    events: EventStore
    checkpoints: CheckpointStore
    swarms: SwarmStore
    memories: MemoryStore
    approvals: ApprovalStore
    idempotency: IdempotencyStore
    capabilities: StorageCapabilities
    # Required, not optional: every run entry point (agent / subagent / swarm
    # worker) persists a RunDefinitionSnapshot so Runtime.resume(child_run_id)
    # can restore its spec + identity after an approval pause. A Storage built
    # without one is rejected at Runtime build time -- resumability is not an
    # opt-in capability.
    run_definitions: RunDefinitionStore
    # Reliable-task store. Optional + None for backward compatibility with
    # existing Storage(...) constructions; TaskRuntime rejects a None tasks
    # store at build time. Backends wire their own (FileStorage -> FileTaskStore).
    tasks: "TaskStore | None" = None
    # Evaluation store. Optional + None for backward compatibility; the eval
    # runner persists lifecycle + results when a caller wires one (e.g. an
    # InMemoryEvalStore or a backend-provided file/SQL store).
    evaluations: "EvalStore | None" = None

    def transaction(self) -> "AsyncIterator[_UnitOfWork]":
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
    """Storage backed by independent file-system backends. Each backend manages
    its own files, so cross-store transactions are NOT available -- transaction()
    raises StorageCapabilityError. Branch on capabilities.cross_store_transactions
    (False here) before calling it."""

    def __init__(self, *, root: "str | Path" = "./data") -> None:
        # Lazy import keeps `import linktools.ai` / `import linktools.ai.storage`
        # from pulling the task/evaluation domains; only constructing a
        # FileStorage does.
        from .file.evaluation import FileEvaluationStore
        from .file.task import FileTaskStore

        root_path = Path(root)
        super().__init__(
            resources=ResourceStore(
                primary=FileResourceBackend(root=root_path / "resources")
            ),
            sessions=FileSessionStore(root=root_path / "sessions"),
            runs=FileRunStore(root=root_path / "runs"),
            events=FileEventStore(root=root_path / "events"),
            checkpoints=FileCheckpointStore(root=root_path / "checkpoints"),
            swarms=FileSwarmStore(root=root_path / "swarms"),
            memories=FileMemoryStore(root=root_path / "memories"),
            approvals=FileApprovalStore(root=root_path / "approvals"),
            idempotency=FileIdempotencyStore(root=root_path / "idempotency"),
            run_definitions=FileRunDefinitionStore(root=root_path / "definitions"),
            capabilities=FILE_STORAGE_CAPABILITIES,
            tasks=FileTaskStore(root_path / "tasks"),
            evaluations=FileEvaluationStore(root_path / "evaluations"),
        )
        # Stash the root so the FileRunCommitCoordinator can place its crash-
        # recovery journal under {root}/transactions (frozen dataclass -> bypass).
        object.__setattr__(self, "_root", root_path)

    @property
    def root(self) -> Path:
        """The storage root directory (where per-store subdirs live)."""
        return self._root
