#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyStorage: the SQLAlchemy-backed Storage composition. Lives in its
own module so the core ``storage`` package (and ``linktools.ai`` itself) imports
cleanly without SQLAlchemy installed -- this module is only reached when a
caller actually requests ``SqlAlchemyStorage``. SQLAlchemy and
aiosqlite are optional dependencies; install via ``linktools-ai[sqlite]``.

All stores share one ``session_factory``; ``transaction()`` yields a
UnitOfWork whose stores bind to one AsyncSession + one transaction so a caller
can coordinate writes across stores atomically."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING

try:  # optional dependency -- give a clear install hint instead of a raw ImportError
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - exercised via import-blocking test
    if exc.name and exc.name.split(".")[0] in {"sqlalchemy", "aiosqlite"}:
        raise ImportError(
            "SqlAlchemyStorage requires optional SQLAlchemy dependencies. "
            "Install with one of:\n"
            "  pip install 'linktools-ai[sqlite]'\n"
            "  pip install 'linktools-ai[sqlalchemy]'"
        ) from exc
    raise

if TYPE_CHECKING:
    from ...artifact.coordination import ArtifactDigestCoordinator
    from ...evaluation.store import EvalStore
    from ...jobs.store import JobStore
    from ..protocols import (
        ArtifactBlobStore,
        ArtifactRecordStore,
        LeaseCoordinator,
        StorageUnitOfWork,
    )

from ...asset.store import AssetStore
from ...agent.approval import ApprovalStore
from ...events.store import EventStore
from ...memory.store import MemoryStore
from ...run.checkpoint import CheckpointStore
from ...run.store import RunStore
from ...session.store import SessionStore
from ...swarm.store import SwarmStore
from ...tool.idempotency import IdempotencyStore
from ..features import SQLALCHEMY_STORAGE_FEATURES, StorageFeatures
from ..facade import Storage
from .approval import SqlAlchemyApprovalStore
from .artifact_record import SqlAlchemyArtifactRecordStore
from .dialects import resolve_dialect_strategy
from .checkpoint import SqlAlchemyCheckpointStore
from .definition import SqlAlchemyRunDefinitionStore
from .event import SqlAlchemyEventStore
from .evaluation import SqlAlchemyEvalStore
from .idempotency import SqlAlchemyIdempotencyStore
from .memory import SqlAlchemyMemoryStore
from .naming import DEFAULT_SQL_NAMING, SqlNamingStrategy
from .asset import SqlAlchemyAssetBackend
from .run import SqlAlchemyRunStore
from .session import SqlAlchemySessionStore
from .swarm import SqlAlchemySwarmStore
from .job import SqlAlchemyJobStore


@dataclass(frozen=True)
class _UnitOfWork:
    """Atomic cross-store unit of work. Yielded by
    SqlAlchemyStorage.transaction(). All stores bind to the SAME AsyncSession,
    and that session's open transaction is owned by the surrounding
    ``async with`` -- writes through tx.runs / tx.approvals / etc. either all
    commit (clean exit) or all roll back (exception). Stores in UoW mode do NOT
    open their own sessions or call session.begin(); they reuse ``session`` and
    flush after each operation so subsequent reads within the unit observe prior
    writes.

    Field types track ``StorageUnitOfWork`` (storage.protocols): the store
    Protocols directly, no ``Any``. ``artifact_records`` is a real
    session-bound ``SqlAlchemyArtifactRecordStore`` sharing this UoW's session
    (flush, not commit) so artifact-record writes join the same atomic scope as
    run/event/job writes. ``assets`` is a real session-bound ``AssetStore``
    whose backend reuses this UoW's session for both reads and writes, so asset
    mutations (put/delete/move, revision, idempotency) commit or roll back with
    every other store in the unit."""

    session: AsyncSession
    assets: AssetStore
    artifact_records: "ArtifactRecordStore"
    runs: RunStore
    events: EventStore
    checkpoints: CheckpointStore
    approvals: ApprovalStore
    sessions: SessionStore
    swarms: SwarmStore
    memories: MemoryStore
    idempotency: IdempotencyStore
    jobs: "JobStore"
    evaluations: "EvalStore"


class SqlAlchemyStorageAdapter(Storage):
    """The generic SQLAlchemy Storage adapter.

    This is the thin, backend-neutral surface a DOWNSTREAM composes: it takes a
    caller-constructed ``async_sessionmaker`` (NEVER a URL/engine), the caller's
    chosen :class:`ArtifactBlobStore` + :class:`LeaseCoordinator` +
    :class:`StorageFeatures`, and wires the SQLAlchemy metadata stores around
    them. It imports no dialect driver, constructs no engine, branches on no
    dialect name, and takes the blob store + coordination + features as INJECTED
    dependencies -- the caller owns those choices. The artifact facade is built
    over the injected blob store + a session-bound SqlAlchemyArtifactRecordStore
    so artifact records share the cross-store transaction.

    The in-repo :class:`SqlAlchemyStorage` and :class:`~..sqlite.SqliteStorage`
    conveniences are thin subclasses that supply default blob/coordination/
    features and delegate here."""

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        artifact_blobs: "ArtifactBlobStore",
        coordination: "LeaseCoordinator | None",
        features: StorageFeatures,
        artifact_coordinator: "ArtifactDigestCoordinator",
        naming: "SqlNamingStrategy" = DEFAULT_SQL_NAMING,
    ) -> None:
        from ...artifact.store import ArtifactStore

        # Apply the naming convention to the shared declarative metadata (the
        # real SQLAlchemy mechanism for constraint/index name derivation; a
        # downstream can standardize them for migration DDL).
        from .models import Base

        if naming.naming_convention:
            Base.metadata.naming_convention = dict(naming.naming_convention)

        # Resolve the dialect strategy eagerly so an unsupported dialect fails
        # at construction rather than on first write. The asset backend uses
        # this strategy to classify integrity violations portably.
        self._dialect_strategy = resolve_dialect_strategy(session_factory)

        assets = AssetStore(
            primary=SqlAlchemyAssetBackend(
                session_factory=session_factory, strategy=self._dialect_strategy
            )
        )
        super().__init__(
            assets=assets,
            sessions=SqlAlchemySessionStore(session_factory=session_factory),
            runs=SqlAlchemyRunStore(session_factory=session_factory),
            events=SqlAlchemyEventStore(session_factory=session_factory),
            checkpoints=SqlAlchemyCheckpointStore(session_factory=session_factory),
            swarms=SqlAlchemySwarmStore(session_factory=session_factory),
            memories=SqlAlchemyMemoryStore(session_factory=session_factory),
            approvals=SqlAlchemyApprovalStore(session_factory=session_factory),
            idempotency=SqlAlchemyIdempotencyStore(session_factory=session_factory),
            run_definitions=SqlAlchemyRunDefinitionStore(
                session_factory=session_factory
            ),
            jobs=SqlAlchemyJobStore(session_factory=session_factory),
            evaluations=SqlAlchemyEvalStore(session_factory=session_factory),
            features=features,
            coordination=coordination,
            _transaction_manager=_SqlAlchemyTransactionManager(session_factory),
            artifacts=ArtifactStore(
                artifact_blobs,
                SqlAlchemyArtifactRecordStore(
                    session_factory=session_factory, strategy=self._dialect_strategy
                ),
                artifact_coordinator,
            ),
        )


class SqlAlchemyStorage(SqlAlchemyStorageAdapter):
    """Convenience SQLAlchemy composition: a caller hands a session_factory +
    a blobs_root and gets process-local coordination, default DATABASE-scope
    features, and Filesystem-backed artifact blobs. Delegates the real wiring
    to :class:`SqlAlchemyStorageAdapter`. For a deployment
    that brings its own blob store / coordination / features, use the adapter
    directly."""

    def __init__(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        blobs_root: Path,
    ) -> None:
        from ..coordination.process_local import ProcessLocalLeaseCoordinator
        from ..filesystem.artifact import FilesystemArtifactBlobStore
        from ..filesystem.artifact_coordination import (
            FilesystemArtifactDigestCoordinator,
        )

        super().__init__(
            session_factory=session_factory,
            artifact_blobs=FilesystemArtifactBlobStore(blobs_root=blobs_root),
            coordination=ProcessLocalLeaseCoordinator(),
            features=SQLALCHEMY_STORAGE_FEATURES,
            # Blobs live on the shared filesystem, so the per-digest lock must
            # span processes (a separate sweeper worker) -- flock the blobs root.
            artifact_coordinator=FilesystemArtifactDigestCoordinator(root=blobs_root),
        )


class _SqlAlchemyTransactionManager:
    """The StorageTransactionManager for SqlAlchemyStorage: yields a UoW whose
    stores all share one AsyncSession + one transaction. ``async with
    session.begin()`` auto-commits on clean exit and auto-rollbacks on
    exception, giving true atomicity across stores: either every tx.* write
    persists, or none of them do. Lives here (next to _UnitOfWork) so the
    manager, the UoW, and the bound-store construction stay in one place; the
    The internal _transaction_manager holds an instance and Storage.transaction()
    delegates to it."""

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]") -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[StorageUnitOfWork]":
        async with self._session_factory() as session:
            async with session.begin():
                yield _UnitOfWork(
                    session=session,
                    # assets: session-bound -- the backend reuses this UoW's
                    # session for reads (no close) and writes (flush only; the
                    # UoW owns begin/commit/rollback), so an asset mutation
                    # commits or rolls back with every other store here.
                    assets=AssetStore(
                        primary=SqlAlchemyAssetBackend(
                            session_factory=self._session_factory, session=session
                        )
                    ),
                    # artifact_records: session-bound -- joins the UoW's atomic
                    # scope (flush, not commit) so an artifact-record write
                    # rolls back with a run/event write on failure.
                    artifact_records=SqlAlchemyArtifactRecordStore(
                        session_factory=self._session_factory, session=session
                    ),
                    runs=SqlAlchemyRunStore(
                        session_factory=self._session_factory, session=session
                    ),
                    events=SqlAlchemyEventStore(
                        session_factory=self._session_factory, session=session
                    ),
                    checkpoints=SqlAlchemyCheckpointStore(
                        session_factory=self._session_factory, session=session
                    ),
                    approvals=SqlAlchemyApprovalStore(
                        session_factory=self._session_factory, session=session
                    ),
                    sessions=SqlAlchemySessionStore(
                        session_factory=self._session_factory, session=session
                    ),
                    swarms=SqlAlchemySwarmStore(
                        session_factory=self._session_factory, session=session
                    ),
                    memories=SqlAlchemyMemoryStore(
                        session_factory=self._session_factory, session=session
                    ),
                    idempotency=SqlAlchemyIdempotencyStore(
                        session_factory=self._session_factory, session=session
                    ),
                    jobs=SqlAlchemyJobStore(
                        session_factory=self._session_factory, session=session
                    ),
                    evaluations=SqlAlchemyEvalStore(
                        session_factory=self._session_factory, session=session
                    ),
                )
