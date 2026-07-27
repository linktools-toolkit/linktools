#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyStorageAdapter: the SQLAlchemy-backed Storage composition. Lives in its
own module so the core ``storage`` package (and ``linktools.ai`` itself) imports
cleanly without SQLAlchemy installed -- this module is only reached when a
caller actually requests the SQLAlchemy adapter. SQLAlchemy and
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
            "SQLAlchemy storage requires optional dependencies. "
            "Install with one of:\n"
            "  pip install 'linktools-ai[sqlite]'\n"
            "  pip install 'linktools-ai[sqlalchemy]'"
        ) from exc
    raise

if TYPE_CHECKING:
    from ...artifact.persistence.protocols import ArtifactRecordStore
    from ...evaluation.store import EvalStore
    from ...jobs.store import JobStore
    from ...storage.blob.protocols import BlobStore
    from ...storage.coordination.protocols import KeyedCoordinator, LeaseCoordinator
    from .protocols import (
        StorageUnitOfWork,
    )

from ...agent.approval import ApprovalStore
from ...artifact.persistence.sqlalchemy import SqlAlchemyArtifactRecordStore
from ...events.store import EventStore
from ...memory.store import MemoryStore
from ...run.checkpoint import CheckpointStore
from ...run.store import RunStore
from ...session.store import SessionStore
from ...swarm.store import SwarmStore
from ...tool.idempotency import IdempotencyStore
from .features import StorageFeatures
from .facade import Storage
from ...agent.persistence.sqlalchemy import SqlAlchemyApprovalStore
from ...storage.sqlalchemy.dialects import SqlAlchemyDialect, resolve_dialect
from ...storage.features import CoordinationScope, StorageComponent, TransactionScope
from ...run.persistence.sqlalchemy.checkpoint import SqlAlchemyCheckpointStore
from ...run.persistence.sqlalchemy.definition import SqlAlchemyRunDefinitionStore
from ...events.persistence.sqlalchemy import SqlAlchemyEventStore
from ...evaluation.persistence.sqlalchemy import SqlAlchemyEvalStore
from ...tool.persistence.sqlalchemy import SqlAlchemyIdempotencyStore
from ...memory.persistence.sqlalchemy import SqlAlchemyMemoryStore
from ...storage.sqlalchemy.naming import DEFAULT_SQL_NAMING, SqlNamingStrategy
from ...storage.backends.sqlalchemy.object import SqlAlchemyObjectBackend
from ...storage.backends.sqlalchemy.schema import SqlAlchemySchemaProvider
from ...storage.object.store import ObjectStore
from ...run.persistence.sqlalchemy.run import SqlAlchemyRunStore
from ...session.persistence.sqlalchemy import SqlAlchemySessionStore
from ...swarm.persistence.sqlalchemy import SqlAlchemySwarmStore
from ...jobs.persistence.sqlalchemy import SqlAlchemyJobStore


@dataclass(frozen=True)
class _UnitOfWork:
    """Atomic cross-store unit of work. Yielded by
    SqlAlchemyStorageAdapter.transaction(). All stores bind to the SAME AsyncSession,
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
    run/event/job writes. ``assets`` is a real session-bound ``ObjectStore``
    whose backend reuses this UoW's session for both reads and writes, so asset
    mutations (put/delete/move, revision, idempotency) commit or roll back with
    every other store in the unit."""

    session: AsyncSession
    assets: ObjectStore
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
    them. It imports no dialect driver, constructs no engine, and takes the
    blob store + coordination + features as INJECTED dependencies -- the
    caller owns those choices. ``dialect`` is optional: when omitted, each
    sub-store lazily auto-detects it from its open session's bound engine on
    first use (see ``storage.sqlalchemy.dialects.resolve_dialect``); the
    adapter itself never branches on the dialect's identity, only calls its
    Protocol methods. The artifact facade is built over the injected blob
    store + a session-bound SqlAlchemyArtifactRecordStore so artifact records
    share the cross-store transaction.

    The in-repo :class:`~..sqlite.SqliteStorage` convenience supplies default
    blob/coordination/features and delegates here."""

    def __init__(
        self,
        *,
        session_factory,
        artifact_blobs,
        coordination,
        features,
        artifact_coordinator,
        dialect=None,
        schema_provider,
    ) -> None:
        raise TypeError("SqlAlchemyStorageAdapter must be created with await create()")

    def _initialize(
        self,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        artifact_blobs: "BlobStore",
        coordination: "LeaseCoordinator | None",
        features: StorageFeatures,
        artifact_coordinator: "KeyedCoordinator",
        dialect: "SqlAlchemyDialect | None" = None,
        schema_provider: "SqlAlchemySchemaProvider",
    ) -> None:
        from ...artifact.store import ArtifactStore

        self._session_factory = session_factory
        # dialect may be left None: each sub-store below auto-detects it from
        # an open session's bound engine on first use. A caller wanting a
        # vendor with no built-in (or a test double) hands its own dialect in
        # instead, threaded to every sub-store so they all agree.
        self._dialect = dialect
        self._schema_provider = schema_provider

        assets = ObjectStore(
            primary=SqlAlchemyObjectBackend(
                session_factory=session_factory, dialect=self._dialect
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
            _transaction_manager=_SqlAlchemyTransactionManager(
                session_factory, dialect=self._dialect
            ),
            artifacts=ArtifactStore(
                artifact_blobs,
                SqlAlchemyArtifactRecordStore(
                    session_factory=session_factory, dialect=self._dialect
                ),
                artifact_coordinator,
            ),
        )

    @classmethod
    async def create(
        cls,
        *,
        session_factory: "async_sessionmaker[AsyncSession]",
        dialect: "SqlAlchemyDialect | None" = None,
        schema_provider: "SqlAlchemySchemaProvider",
        artifact_blobs: "BlobStore",
        coordination: "LeaseCoordinator | None",
        artifact_coordinator: "KeyedCoordinator",
    ) -> "SqlAlchemyStorageAdapter":
        await schema_provider.validate(session_factory)
        instance = cls.__new__(cls)
        instance._initialize(
            session_factory=session_factory,
            artifact_blobs=artifact_blobs,
            coordination=coordination,
            features=StorageFeatures.from_components(
                # Placeholder: Storage.__post_init__ re-derives from the REAL
                # wired objects (transaction manager / coordination / artifacts)
                # once they are set on the instance.
                transaction_manager=_SqlAlchemyTransactionManager(session_factory, dialect=dialect),
                coordination=coordination,
                artifacts=None,
                components={},
            ),
            artifact_coordinator=artifact_coordinator,
            dialect=dialect,
            schema_provider=schema_provider,
        )
        return instance


class _ReferenceSqlAlchemyComposition(SqlAlchemyStorageAdapter):
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
        from ...storage.coordination.file import FilesystemKeyedCoordinator
        from ...storage.coordination.process_local import ProcessLocalLeaseCoordinator
        from ...storage.filesystem.artifact import FilesystemArtifactBlobStore
        from ...storage.backends.sqlalchemy.schema import SqliteReferenceSchemaProvider
        # The commit-log ORM models (run_commit_log, swarm_commit_log) are
        # domain-owned and must be registered on the shared DomainBase metadata
        # BEFORE the schema provider runs CREATE TABLE. Importing them here
        # (the composition root, where domain imports are legal) triggers that
        # registration; the storage kernel's schema provider does NOT import
        # domain packages (architecture boundary: storage never depends on
        # run/swarm).
        from ...run.persistence.sqlalchemy import commit_log as _  # noqa: F401
        from ...swarm.persistence import sqlalchemy_commit as _sw  # noqa: F401

        coordination = ProcessLocalLeaseCoordinator()
        self._initialize(
            session_factory=session_factory,
            artifact_blobs=FilesystemArtifactBlobStore(blobs_root=blobs_root),
            coordination=coordination,
            features=StorageFeatures.from_components(
                transaction_manager=_SqlAlchemyTransactionManager(session_factory),
                coordination=coordination,
                artifacts=None,
                components={},
            ),
            # Blobs live on the shared filesystem, so the per-digest lock must
            # span processes (a separate sweeper worker) -- flock the blobs root.
            artifact_coordinator=FilesystemKeyedCoordinator(root=blobs_root),
            # dialect is omitted: this reference convenience is always handed
            # a SQLite session_factory, so auto-detection resolves it. A
            # downstream that brings its own engine constructs
            # SqlAlchemyStorageAdapter directly (auto-detection works there
            # too) or passes an explicit dialect for a vendor with no built-in.
            schema_provider=SqliteReferenceSchemaProvider(),
        )


class _SqlAlchemyTransactionManager:
    """The StorageTransactionManager for SqlAlchemyStorageAdapter: yields a UoW whose
    stores all share one AsyncSession + one transaction. ``async with
    session.begin()`` auto-commits on clean exit and auto-rollbacks on
    exception, giving true atomicity across stores: either every tx.* write
    persists, or none of them do. Lives here (next to _UnitOfWork) so the
    manager, the UoW, and the bound-store construction stay in one place; the
    The internal _transaction_manager holds an instance and Storage.transaction()
    delegates to it."""

    # The cross-store UoW is a real DATABASE-scope atomic transaction (one
    # AsyncSession + one begin/commit/rollback wrapping every store).
    scope = TransactionScope.DATABASE

    def __init__(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
        *,
        dialect=None,
    ) -> None:
        self._session_factory = session_factory
        # May be None here (auto-detect on first use in transaction()); a
        # caller wanting a vendor with no built-in (or a test double) hands
        # an explicit dialect in instead.
        self._dialect = dialect

    @asynccontextmanager
    async def transaction(self) -> "AsyncIterator[StorageUnitOfWork]":
        from ...storage.backends.sqlalchemy.object import _SqlAlchemyTransactionBackend

        async with self._session_factory() as session:
            # Memoize: resolve_dialect(session, self._dialect) is a no-op
            # once self._dialect is set (explicit override or a prior
            # resolution), so later transactions skip the lookup entirely.
            self._dialect = dialect = resolve_dialect(session, self._dialect)
            async with session.begin():
                yield _UnitOfWork(
                    session=session,
                    # assets: session-bound via the transaction-bound child
                    # backend -- the child reuses this UoW's session for reads
                    # (no close) and writes (flush only; the UoW owns
                    # begin/commit/rollback), so an asset mutation commits or
                    # rolls back with every other store here. The parent
                    # backend's ``session=`` shortcut was removed because it
                    # leaked ambient state; the child IS the session-bound form.
                    assets=ObjectStore(
                        primary=_SqlAlchemyTransactionBackend(
                            session=session,
                            dialect=dialect,
                        )
                    ),
                    # artifact_records: session-bound -- joins the UoW's atomic
                    # scope (flush, not commit) so an artifact-record write
                    # rolls back with a run/event write on failure.
                    artifact_records=SqlAlchemyArtifactRecordStore(
                        session_factory=self._session_factory,
                        session=session,
                        dialect=dialect,
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
