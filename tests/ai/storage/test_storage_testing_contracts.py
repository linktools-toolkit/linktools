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
* JobStoreContract -> FilesystemStorage.tasks (FilesystemTaskStore).
* StorageTransactionManagerContract -> BOTH SqlAlchemyStorage.transactions
  (the supported cross-store UoW path -- skipped when SQLAlchemy is not
  installed) AND NoCrossStoreTransactions (the unsupported-scope path that
  must raise StorageTransactionNotSupportedError at the call).

Mirrors tests/ai/storage/test_external_adapter_conformance.py, which wires
the existing three Contracts against an in-memory external adapter.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.storage.testing import (
    AssetStoreContract,
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
    EventStoreContract,
    JobStoreContract,
    LeaseCoordinatorContract,
    StorageTransactionManagerContract,
)


def test_public_testkit_exports_all_seven_contracts() -> None:
    """The C5 closure: the public testkit surface must expose all seven
    Contracts so a downstream adapter can subclass each one and run the
    contract suite in its own CI."""
    from linktools.ai.storage import testing

    expected = {
        "AssetStoreContract",
        "ArtifactBlobStoreContract",
        "ArtifactRecordStoreContract",
        "EventStoreContract",
        "JobStoreContract",
        "LeaseCoordinatorContract",
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
    """JobStoreContract against FilesystemStorage.tasks -- the in-repo
    reference FilesystemTaskStore."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        from linktools.ai.storage.facade import FilesystemStorage

        self._storage = FilesystemStorage(root=tmp_path)

    def job_store(self):
        return self._storage.tasks


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
        from .example_external_adapter_full import InMemoryEventStore

        return InMemoryEventStore()


class TestExternalJobStoreConformance(JobStoreContract):
    """JobStoreContract against the in-memory external adapter's
    InMemoryJobStore. The adapter is built ONLY from public Protocols, the
    typed JobStore error surface, and the jobs domain models; running the
    same Contract the Filesystem backend passes against it proves the public
    JobStore surface is sufficient for a downstream adapter."""

    def job_store(self):
        from .example_external_adapter_full import InMemoryJobStore

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
    return SqlAlchemyStorage(session_factory=session_factory)


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
    """JobStoreContract against SqlAlchemyStorage.tasks -- the DB-backed
    SqlAlchemyTaskStore. Mirrors the Filesystem reference's fencing-token /
    transition semantics; the contract verifies a SQL backend honors the same
    reliable-task guarantees."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def job_store(self):
        return self._storage.tasks


class TestSqlAlchemyTransactionManagerConformance(StorageTransactionManagerContract):
    """The supported-scope path: SqlAlchemyStorage.transactions yields a real
    UnitOfWork whose stores share one AsyncSession + one transaction. A clean
    exit commits; an exception rolls back with no partial commit."""

    @pytest.fixture(autouse=True)
    def _build_storage(self, tmp_path) -> None:
        self._storage = _build_sqlalchemy_storage(tmp_path)

    def transaction_manager(self):
        return self._storage.transactions

    async def _write_inside_scope(self, uow) -> None:
        from linktools.ai.session.models import SessionRecord, SessionStatus

        now = datetime.now(timezone.utc)
        await uow.sessions.create(
            SessionRecord(
                id="tx-rollback",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

    async def _verify_rollback(self) -> None:
        # After the scope rolled back, the row written inside the scope must
        # not be visible through a fresh query.
        fetched = await self._storage.sessions.get("tx-rollback")
        assert fetched is None
