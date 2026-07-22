#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive the public storage conformance testkit against the in-repo reference
backends so a regression in either the Contracts or the backends is caught
here, not in a downstream adapter's CI.

This module wires the four Contracts added for audit closure C5
(AssetStoreContract, EventStoreContract, JobStoreContract,
StorageTransactionManagerContract) against the in-repo reference backends:

* AssetStoreContract -> FilesystemStorage.assets (FileAssetBackend under an
  AssetStore primary+overlay composition).
* EventStoreContract -> FilesystemStorage.events (FilesystemEventStore).
* JobStoreContract -> FilesystemStorage.jobs (FilesystemJobStore).
* StorageTransactionManagerContract -> BOTH SqlAlchemyStorage._transaction_manager
  (the supported cross-store UoW path -- skipped when SQLAlchemy is not
  installed) AND NoCrossStoreTransactions (the unsupported-scope path that
  must raise StorageTransactionNotSupportedError at the call).

Mirrors tests/ai/storage/test_external_adapter_conformance.py, which wires
the existing three Contracts against an in-memory external adapter.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.testing import (
    AssetStoreContract,
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
    EventStoreContract,
    JobStoreContract,
    LeaseCoordinatorContract,
    StorageFeaturesContract,
    StorageTransactionManagerContract,
)


def test_public_testkit_exports_all_eight_contracts() -> None:
    """The testkit surface must expose all eight Contracts so a downstream
    adapter can subclass each one and run the contract suite in its own CI.
    The eighth, StorageFeaturesContract, pins feature self-consistency: a
    Storage's declared StorageFeatures must match what its stores actually
    support (transactions=DATABASE yields a real UoW; NONE raises)."""
    import linktools.ai.testing as testing

    expected = {
        "AssetStoreContract",
        "ArtifactBlobStoreContract",
        "ArtifactRecordStoreContract",
        "EventStoreContract",
        "JobStoreContract",
        "LeaseCoordinatorContract",
        "StorageFeaturesContract",
        "StorageTransactionManagerContract",
    }
    assert expected.issubset(set(dir(testing)))
    assert set(testing.__all__) == expected


class TestAssetStoreConformance(AssetStoreContract):
    """AssetStoreContract against FilesystemStorage.assets -- the in-repo
    reference FileAssetBackend under the primary+overlay AssetStore
    composition."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        from linktools.ai.storage.facade import FilesystemStorage

        self._storage = FilesystemStorage(root=tmp_path)

    def asset_store(self):
        return self._storage.assets


class TestEventStoreConformance(EventStoreContract):
    """EventStoreContract against FilesystemStorage.events -- the in-repo
    reference FilesystemEventStore."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        from linktools.ai.storage.facade import FilesystemStorage

        self._storage = FilesystemStorage(root=tmp_path)

    def event_store(self):
        return self._storage.events


class TestJobStoreConformance(JobStoreContract):
    """JobStoreContract against FilesystemStorage.jobs -- the in-repo
    reference FilesystemJobStore."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        from linktools.ai.storage.facade import FilesystemStorage

        self._storage = FilesystemStorage(root=tmp_path)

    def job_store(self):
        return self._storage.jobs


# --- additional parameterizations -----------------------------------------
# Each Contract runs against every in-repo backend that implements the store
# type: Memory + Filesystem + SqlAlchemy for assets; Filesystem + SqlAlchemy +
# the external in-memory adapter for events and jobs. A backend that fails the
# contract here is a backend regression, not a contract bug.

class TestMemoryAssetStoreConformance(AssetStoreContract):
    """AssetStoreContract against AssetStore(primary=MemoryAssetBackend()) --
    the in-repo in-memory AssetBackend. Each factory call returns a fresh,
    empty AssetStore so a test cannot observe state written by an earlier
    one."""

    def asset_store(self):
        from linktools.ai.asset.memory import MemoryAssetBackend
        from linktools.ai.asset.store import AssetStore

        return AssetStore(primary=MemoryAssetBackend())


class TestExternalEventStoreConformance(EventStoreContract):
    """EventStoreContract against the in-memory external adapter's
    InMemoryEventStore. The adapter is built ONLY from public Protocols and
    domain models (see example_external_adapter_full); running the same
    Contract the Filesystem backend passes against it proves the public
    EventStore surface is sufficient for a downstream adapter."""

    def event_store(self):
        from external_adapter import InMemoryEventStore

        return InMemoryEventStore()


class TestExternalJobStoreConformance(JobStoreContract):
    """JobStoreContract against the in-memory external adapter's
    InMemoryJobStore. The adapter is built ONLY from public Protocols, the
    typed JobStore error surface, and the jobs domain models; running the
    same Contract the Filesystem backend passes against it proves the public
    JobStore surface is sufficient for a downstream adapter."""

    def job_store(self):
        from external_adapter import InMemoryJobStore

        return InMemoryJobStore()


class TestNoCrossStoreTransactionsConformance(StorageTransactionManagerContract):
    """The unsupported-scope path: NoCrossStoreTransactions.transaction()
    must raise StorageTransactionNotSupportedError AT THE CALL (not later
    inside __aenter__). This is the always-installable proof -- no optional
    SQLAlchemy dependency required."""

    def is_supported(self) -> bool:
        return False

    def transaction_manager(self):
        from linktools.ai.storage.transaction import NoCrossStoreTransactions

        return NoCrossStoreTransactions(backend_name="TestBackend")


class TestFilesystemStorageFeaturesConformance(StorageFeaturesContract):
    """StorageFeaturesContract against FilesystemStorage. The Filesystem
    declares transactions=NONE (each file store is independently durable, no
    cross-store UoW), so transaction() must raise at the call. streaming_blobs
    is True, so artifacts MUST be wired. Pins the feature self-consistency
    contract against the in-repo file reference."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        from linktools.ai.storage.facade import FilesystemStorage

        self._storage = FilesystemStorage(root=tmp_path)

    def storage(self):
        return self._storage


# --- SqlAlchemy cross-store UoW path --------------------------------------
# Skipped entirely when SQLAlchemy/aiosqlite is not installed. The supported
# path is exercised here; the unsupported path is covered above without any
# optional dependencies.

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")


def _build_sqlalchemy_storage(tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.storage import SqlAlchemyStorage
    from linktools.ai.storage.sqlalchemy.models import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/contract.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/contract.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )


class TestSqlAlchemyAssetStoreConformance(AssetStoreContract):
    """AssetStoreContract against SqlAlchemyStorage.assets -- the DB-backed
    SqlAlchemyAssetBackend under the AssetStore primary composition. Shares
    the same engine/session_factory pattern as the transaction contract so a
    regression in the SQL asset path is caught here, not only in downstream
    adapter CI."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def asset_store(self):
        return self._storage.assets


class TestSqlAlchemyEventStoreConformance(EventStoreContract):
    """EventStoreContract against SqlAlchemyStorage.events -- the DB-backed
    SqlAlchemyEventStore. The store owns sequence assignment (server-side
    MAX+1); the contract verifies the same append/list semantics the
    Filesystem backend provides."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def event_store(self):
        return self._storage.events


class TestSqlAlchemyJobStoreConformance(JobStoreContract):
    """JobStoreContract against SqlAlchemyStorage.jobs -- the DB-backed
    SqlAlchemyJobStore. Mirrors the Filesystem reference's fencing-token /
    transition semantics; the contract verifies a SQL backend honors the same
    reliable-task guarantees."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def job_store(self):
        return self._storage.jobs


class TestSqlAlchemyTransactionManagerConformance(StorageTransactionManagerContract):
    """The supported-scope path: SqlAlchemyStorage._transaction_manager yields a real
    UnitOfWork whose stores share one AsyncSession + one transaction. A clean
    exit commits; an exception rolls back with no partial commit.

    The contract's default ``_write_inside_scope`` / ``_verify_rollback``
    pair already probes via ``uow.sessions`` through a fresh transaction, so
    no override is needed -- the default IS the assertion a real rollback
    undoes writes (the row written inside the rolled-back scope must be
    absent through a subsequent scope)."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def transaction_manager(self):
        return self._storage._transaction_manager


class TestSqlAlchemyStorageFeaturesConformance(StorageFeaturesContract):
    """StorageFeaturesContract against SqlAlchemyStorage. The SQL backend
    declares transactions=DATABASE (a real cross-store UoW), so
    transaction() must yield a real UoW (not raise). streaming_blobs is True,
    so artifacts MUST be wired. Pins the feature self-consistency contract
    against the in-repo SQL reference."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def storage(self):
        return self._storage
