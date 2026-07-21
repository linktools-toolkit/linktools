#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage facade: composes the storage
backends into one frozen dataclass so a caller gets a single object that can do
everything.

This module is deliberately SQLAlchemy-free: ``Storage`` and
``FilesystemStorage`` depend only on the standard library and core stores, so
``import linktools.ai`` and ``import linktools.ai.storage`` succeed without the
optional SQLAlchemy/aiosqlite dependencies. The SQLAlchemy-backed composition
(``SqlAlchemyStorage``) lives in ``linktools.ai.storage.sqlalchemy`` and is
loaded lazily via ``storage/__init__.__getattr__``.

- Storage: frozen composition of the nine backends + capabilities; the base
  ``transaction()`` delegates to the ``transactions`` field. A backend whose
  stores share a transaction provider (SqlAlchemyStorage) yields a real UoW; a
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
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ...evaluation.store import EvalStore
    from ..artifact.store import ArtifactStore
    from ..jobs.store import JobStore
    from .protocols import (
        LeaseCoordinator,
        StorageTransactionManager,
        StorageUnitOfWork,
    )

from ..agent.approval import ApprovalStore
from ..events.store import EventStore
from ..memory.store import MemoryStore
from ..run.checkpoint import CheckpointStore
from ..run.definition import RunDefinitionStore
from ..run.store import RunStore
from ..session.store import SessionStore
from ..swarm.store import SwarmStore
from ..tool.idempotency import IdempotencyStore
from .features import FILE_STORAGE_FEATURES, StorageFeatures
from .filesystem.approval import FilesystemApprovalStore
from .filesystem.checkpoint import FilesystemCheckpointStore
from .filesystem.definition import FilesystemRunDefinitionStore
from .filesystem.event import FilesystemEventStore
from .filesystem.idempotency import FilesystemIdempotencyStore
from .filesystem.memory import FilesystemMemoryStore
from .filesystem.run import FilesystemRunStore
from .filesystem.session import FilesystemSessionStore
from .filesystem.swarm import FilesystemSwarmStore
from ..asset.file import FileAssetBackend
from ..asset.store import AssetStore


@dataclass(frozen=True)
class Storage:
    """Frozen composition of the storage backends. Concrete subclasses
    (FilesystemStorage, SqlAlchemyStorage) are responsible for constructing the
    backends; this base only holds them and exposes the cross-cutting
    transaction() hook."""

    assets: AssetStore
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
    # configured against process-local coordination is part of the later
    # RuntimeRequirements work (not yet wired); until then Storage.coordination
    # is an available, conformant capability without an enforcing consumer.
    coordination: "LeaseCoordinator"
    # Cross-store UnitOfWork manager. The canonical surface for an
    # atomic cross-store scope: storage.transactions.transaction() yields a UoW
    # whose stores share one transaction. A backend whose stores are independent
    # (FilesystemStorage) supplies a NoCrossStoreTransactions manager that
    # raises StorageTransactionNotSupportedError at the call -- the honest
    # declaration for features.transactions = PROCESS_LOCAL (single-store
    # durability only, no cross-store UoW). A backend with a shared transaction
    # provider (SqlAlchemyStorage) supplies a real manager and declares
    # TransactionScope.DATABASE.
    transactions: "StorageTransactionManager"
    # Required, not optional: every run entry point (agent / subagent / swarm
    # worker) persists a RunDefinitionSnapshot so Runtime.resume(child_run_id)
    # can restore its spec + identity after an approval pause. A Storage built
    # without one is rejected at Runtime build time -- resumability is not an
    # opt-in capability.
    run_definitions: RunDefinitionStore
    # Reliable-task store (jobs domain). Optional + None for backward
    # compatibility with existing Storage(...) constructions; JobRuntime rejects
    # a None jobs store at build time. Backends wire their own
    # (FilesystemStorage -> FilesystemJobStore).
    jobs: "JobStore | None" = None
    # Evaluation store. Optional + None for backward compatibility; the eval
    # runner persists lifecycle + results when a caller wires one (e.g. an
    # InMemoryEvalStore or a backend-provided file/SQL store).
    evaluations: "EvalStore | None" = None
    # Artifact store (content-addressed blobs + lineage records). Optional +
    # None for backward compatibility; backends wire one and JobRuntime consumes
    # it explicitly (no implicit getattr fallback on the asset store).
    artifacts: "ArtifactStore | None" = None

    def transaction(self) -> "AsyncIterator[StorageUnitOfWork]":
        """Cross-store transactional scope, delegating to the ``transactions``
        field (the canonical surface). A backend whose stores share a
        transaction provider (SqlAlchemyStorage) yields a real UoW; a backend
        whose stores are independent (FilesystemStorage) raises
        StorageTransactionNotSupportedError at the call. Branch on
        ``features.transactions is TransactionScope.DATABASE`` before relying on
        it. Retained as a thin delegator so existing call sites (run commit
        coordinators, tests) keep working; new code should call
        ``storage.transactions.transaction()`` directly."""
        return self.transactions.transaction()


class FilesystemStorage(Storage):
    """Storage backed by independent file-system backends. Each backend manages
    its own files, so cross-store transactions are NOT available -- transaction()
    raises StorageTransactionNotSupportedError. Branch on features.transactions
    == TransactionScope.DATABASE (False here) before calling it."""

    def __init__(self, *, root: "str | Path" = "./data") -> None:
        # Lazy import keeps `import linktools.ai` / `import linktools.ai.storage`
        # from pulling the jobs/evaluation domains; only constructing a
        # FilesystemStorage does.
        from .filesystem.evaluation import FilesystemEvaluationStore
        from .filesystem.job import FilesystemJobStore

        root_path = Path(root)
        from ..artifact.store import ArtifactStore
        from .coordination.process_local import ProcessLocalLeaseCoordinator
        from .filesystem.artifact import (
            FilesystemArtifactBlobStore,
            FilesystemArtifactRecordStore,
        )
        from .transaction import NoCrossStoreTransactions

        assets = AssetStore(primary=FileAssetBackend(root=root_path / "resources"))
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
            features=FILE_STORAGE_FEATURES,
            coordination=ProcessLocalLeaseCoordinator(),
            transactions=NoCrossStoreTransactions("FilesystemStorage"),
            jobs=FilesystemJobStore(root_path / "jobs"),
            evaluations=FilesystemEvaluationStore(root=root_path / "evaluations"),
            artifacts=ArtifactStore(
                FilesystemArtifactBlobStore(
                    blobs_root=root_path / "artifacts" / "blobs"
                ),
                FilesystemArtifactRecordStore(
                    records_root=root_path / "artifacts" / "records"
                ),
            ),
        )
        # Stash the root so the FilesystemRunCommitCoordinator can place its crash-
        # recovery journal under {root}/transactions (frozen dataclass -> bypass).
        object.__setattr__(self, "_root", root_path)

    @property
    def root(self) -> Path:
        """The storage root directory (where per-store subdirs live)."""
        return self._root
