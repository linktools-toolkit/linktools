#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence facade: composes each domain's persistence/ adapters
into one frozen ``Storage`` dataclass so a caller gets a single object that
can do everything.

Lives at the runtime layer (NOT under ``storage/``) so the storage kernel
stays free of any domain import: ``import linktools.ai.storage`` pulls only
the storage-kernel (object / cache / blob / coordination / backends) and
never any domain package. Each domain owns its persistence/ subpackage; this
facade composes those adapters + the storage-kernel ObjectStore.

This module is deliberately SQLAlchemy-free: ``Storage`` and
``FilesystemStorage`` depend only on the standard library and core stores, so
``import linktools.ai`` and ``import linktools.ai.runtime`` succeed without
the optional SQLAlchemy/aiosqlite dependencies. The SQLAlchemy-backed
composition (``SqlAlchemyStorageAdapter``) lives at
``linktools.ai.runtime.persistence.sqlalchemy`` and is loaded lazily via
``runtime/persistence/__init__.__getattr__``.

- Storage: frozen composition of the nine backends + capabilities; the base
  ``transaction()`` delegates to the internal ``_transaction_manager``. A backend whose
  stores share a transaction provider (SqlAlchemyStorageAdapter) yields a real UoW; a
  backend whose stores are independent (FilesystemStorage) raises
  StorageTransactionNotSupportedError at the call.
- FilesystemStorage: nine independent file backends under a root dir. No cross-store
  transactions are possible, so the inherited transaction() raises
  StorageTransactionNotSupportedError.

Subclasses use object.__setattr__ to stash their own state (e.g. the session
factory) because the dataclass is frozen -- hence frozen=True rather than
slots=True, which would also forbid per-subclass attributes."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Mapping

if TYPE_CHECKING:
    from ...artifact.persistence.protocols import ArtifactRecordStore
    from ...artifact.store import ArtifactStore
    from ...evaluation.store import EvalStore
    from ...jobs.store import JobStore
    from ...storage.coordination.protocols import LeaseCoordinator
    from .protocols import (
        StorageTransactionManager,
        StorageUnitOfWork,
    )

from ...errors import StorageCapabilityError
from ...agent.approval import ApprovalStore
from ...events.store import EventStore
from ...memory.store import MemoryStore
from ...run.checkpoint import CheckpointStore
from ...run.definition import RunDefinitionStore
from ...run.store import RunStore
from ...session.store import SessionStore
from ...storage.object.store import ObjectStore
from ...swarm.store import SwarmStore
from ...tool.idempotency import IdempotencyStore
from .features import StorageFeatures
from ...storage.features import CoordinationScope, StorageComponent, TransactionScope
from .transaction import NoCrossStoreTransactions


def storage_component_map(storage: "Storage") -> "Mapping[StorageComponent, object | None]":
    """The single source of truth for which wired object backs each
    StorageComponent. Used by both ``Storage.__post_init__`` (to derive
    features) and the consistency gate (to cross-check declared vs wired) so
    the two lists cannot drift. Optional components (jobs / artifacts) are
    omitted when their object is None -- a derived feature set never contains
    a capability for an unwired component."""
    components: "dict[StorageComponent, object | None]" = {
        StorageComponent.ASSETS: storage.assets,
        StorageComponent.RUNS: storage.runs,
        StorageComponent.SESSIONS: storage.sessions,
        StorageComponent.EVENTS: storage.events,
        StorageComponent.APPROVALS: storage.approvals,
        StorageComponent.CHECKPOINTS: storage.checkpoints,
        StorageComponent.SWARMS: getattr(storage, "swarms", None),
        StorageComponent.MEMORIES: getattr(storage, "memories", None),
        StorageComponent.IDEMPOTENCY: getattr(storage, "idempotency", None),
    }
    if storage.jobs is not None:
        components[StorageComponent.JOBS] = storage.jobs
    if storage.artifacts is not None:
        # ARTIFACT_RECORDS is the artifact domain's record store, surfaced
        # through the ArtifactStore facade. Its capabilities feed
        # transactional_components (SQL shares the UoW session).
        components[StorageComponent.ARTIFACT_RECORDS] = storage.artifacts.record_store
    if getattr(storage, "evaluations", None) is not None:
        components[StorageComponent.EVALUATIONS] = storage.evaluations
    return components


@dataclass(frozen=True)
class Storage:
    """Frozen composition of the storage backends. Concrete subclasses
    (FilesystemStorage, SqliteStorage) are responsible for constructing the
    backends; this base only holds them and exposes the cross-cutting
    transaction() hook."""

    # "assets" is just the business name for this object-storage composition --
    # the field is not tied to any particular backend type.
    assets: ObjectStore
    sessions: SessionStore
    runs: RunStore
    events: EventStore
    checkpoints: CheckpointStore
    swarms: SwarmStore
    memories: MemoryStore
    approvals: ApprovalStore
    idempotency: IdempotencyStore
    features: StorageFeatures
    # Lease coordination. The in-repo references ship a process-local
    # LeaseCoordinator (ProcessLocalLeaseCoordinator); a deployment needing multi-worker
    # Jobs or multi-process Swarms injects a distributed one and declares
    # CoordinationScope.DISTRIBUTED on its StorageFeatures. The build-time
    # capability gate that REJECTS a multi-worker/multi-process topology
    # configured against process-local coordination is enforced by the build-time
    # capability gate (run.requirements); Storage.coordination is the available,
    # conformant capability that gate reads.
    coordination: "LeaseCoordinator"
    # Cross-store UnitOfWork manager. INTERNAL: callers go through
    # storage.transaction(), never the manager object directly. A backend whose
    # stores are independent (FilesystemStorage) supplies a
    # NoCrossStoreTransactions manager that raises
    # StorageTransactionNotSupportedError at the call -- the honest declaration
    # for features.transaction_scope = NONE (each store independently durable,
    # no cross-store UoW). A backend with a shared transaction provider
    # (SqlAlchemyStorageAdapter) supplies a real manager and declares
    # TransactionScope.DATABASE.
    _transaction_manager: "StorageTransactionManager"
    # Required, not optional: every run entry point (agent / subagent / swarm
    # worker) persists a RunDefinitionSnapshot so Runtime.resume(child_run_id)
    # can restore its spec + identity after an approval pause. A Storage built
    # without one is rejected at Runtime build time -- resumability is not an
    # opt-in capability.
    run_definitions: RunDefinitionStore
    # Reliable-task store (jobs domain). Optional + None for existing
    # Storage(...) constructions; JobRuntime rejects
    # a None jobs store at build time. Backends wire their own
    # (FilesystemStorage -> FilesystemJobStore).
    jobs: "JobStore | None" = None
    # Evaluation store. Optional + None when a caller does not wire one; the
    # eval runner persists lifecycle + results when a backend supplies it.
    evaluations: "EvalStore | None" = None
    # Artifact store (content-addressed blobs + lineage records). Optional +
    # None when a backend does not supply one; JobRuntime consumes it explicitly
    # when present (no implicit getattr fallback on the asset store).
    artifacts: "ArtifactStore | None" = None

    def __post_init__(self) -> None:
        """Derive public capabilities from the REAL wired objects (transaction
        manager, coordinator, artifact store, each component store). A Storage
        that is not self-consistent (declares a capability none of its wired
        objects provides) fails construction -- the consistency gate runs
        immediately, not only in RuntimeBuilder."""
        components = storage_component_map(self)
        object.__setattr__(
            self,
            "features",
            StorageFeatures.from_components(
                transaction_manager=self._transaction_manager,
                coordination=self.coordination,
                artifacts=self.artifacts,
                components=components,
            ),
        )
        from .validation import validate_storage_feature_consistency

        validate_storage_feature_consistency(self)

    @property
    def artifact_records(self) -> "ArtifactRecordStore":
        """The wired ArtifactRecordStore (the ARTIFACT_RECORDS component),
        surfaced through the ArtifactStore facade. Raises if no ArtifactStore
        is wired -- callers branch on ``features.streaming_artifacts`` first."""
        if self.artifacts is None:
            raise StorageCapabilityError(
                "this Storage has no ArtifactStore wired; artifact_records "
                "is unavailable"
            )
        return self.artifacts.record_store

    def transaction(self) -> "AsyncIterator[StorageUnitOfWork]":
        """Cross-store transactional scope -- the single public entry. A backend
        whose stores share a transaction provider (SqlAlchemyStorageAdapter) yields a
        real UoW; a backend whose stores are independent (FilesystemStorage)
        raises StorageTransactionNotSupportedError at the call. Branch on
        ``features.transaction_scope is TransactionScope.DATABASE`` before
        relying on it. The wired transaction manager is an internal dependency;
        callers go through this method, never the manager object directly."""
        return self._transaction_manager.transaction()

    @property
    def transaction_manager(self) -> "StorageTransactionManager":
        return self._transaction_manager


class FilesystemStorage(Storage):
    """Storage backed by independent file-system backends. Each backend manages
    its own files, so cross-store transactions are NOT available -- transaction()
    raises StorageTransactionNotSupportedError. Branch on
    features.transaction_scope == TransactionScope.DATABASE (False here) before
    calling it."""

    def __init__(
        self,
        *,
        root: "str | Path" = "./data",
        mode: "FilesystemSecurityMode | None" = None,
    ) -> None:
        # Lazy import keeps `import linktools.ai` / `import linktools.ai.runtime`
        # from pulling the per-domain persistence adapters eagerly; only
        # constructing a FilesystemStorage does.
        from ...agent.persistence.filesystem import FilesystemApprovalStore
        from ...artifact.persistence.filesystem import FilesystemArtifactRecordStore
        from ...artifact.store import ArtifactStore
        from ...events.persistence.filesystem import FilesystemEventStore
        from ...evaluation.persistence.filesystem import FilesystemEvaluationStore
        from ...jobs.persistence.filesystem import FilesystemJobStore
        from ...memory.persistence.filesystem import FilesystemMemoryStore
        from ...run.persistence.checkpoint import FilesystemCheckpointStore
        from ...run.persistence.definition import FilesystemRunDefinitionStore
        from ...run.persistence.run import FilesystemRunStore
        from ...session.persistence.filesystem import FilesystemSessionStore
        from ...storage.backends.filesystem.object import FilesystemObjectBackend
        from ...storage.backends.filesystem.secure_directory import FilesystemSecurityMode
        from ...storage.coordination.process_local import (
            InProcessKeyedCoordinator,
            ProcessLocalLeaseCoordinator,
        )
        from ...storage.filesystem.artifact import FilesystemArtifactBlobStore
        from ...swarm.persistence.filesystem import FilesystemSwarmStore
        from ...tool.persistence.filesystem import FilesystemIdempotencyStore

        root_path = Path(root)
        assets = ObjectStore(
            primary=FilesystemObjectBackend(
                root=root_path / "assets",
                mode=mode or FilesystemSecurityMode.SECURE_POSIX,
            )
        )
        artifact_store = ArtifactStore(
            FilesystemArtifactBlobStore(
                blobs_root=root_path / "artifacts" / "blobs"
            ),
            FilesystemArtifactRecordStore(
                records_root=root_path / "artifacts" / "records"
            ),
            InProcessKeyedCoordinator(),
        )
        coordination = ProcessLocalLeaseCoordinator()
        transaction_manager = NoCrossStoreTransactions("FilesystemStorage")
        super().__init__(
            assets=assets,
            sessions=FilesystemSessionStore(root=root_path / "sessions"),
            runs=FilesystemRunStore(root=root_path / "runs"),
            events=FilesystemEventStore(root=root_path / "events"),
            checkpoints=FilesystemCheckpointStore(root=root_path / "checkpoints"),
            swarms=FilesystemSwarmStore(root=root_path / "swarms"),
            memories=FilesystemMemoryStore(root=root_path / "memories"),
            approvals=FilesystemApprovalStore(root=root_path / "approvals"),
            idempotency=FilesystemIdempotencyStore(root=root_path / "idempotency"),
            run_definitions=FilesystemRunDefinitionStore(root=root_path / "definitions"),
            features=StorageFeatures.from_components(
                transaction_manager=transaction_manager,
                coordination=coordination,
                artifacts=artifact_store,
                components={StorageComponent.ASSETS: assets},
            ),
            coordination=coordination,
            _transaction_manager=transaction_manager,
            jobs=FilesystemJobStore(root_path / "jobs"),
            evaluations=FilesystemEvaluationStore(root=root_path / "evaluations"),
            artifacts=artifact_store,
        )
        # Stash the root so the FilesystemRunCommitCoordinator can place its crash-
        # recovery journal under {root}/transactions (frozen dataclass -> bypass).
        object.__setattr__(self, "_root", root_path)

    @property
    def root(self) -> Path:
        """The storage root directory (where per-store subdirs live)."""
        return self._root


__all__: "list[str]" = [
    "FilesystemStorage",
    "Storage",
    "StorageFeatures",
    "StorageTransactionManager",
    "StorageUnitOfWork",
]
