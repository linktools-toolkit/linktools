#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageFeatures: the component-level capability surface each backend
declares. transaction/coordination are scopes (none/process_local/database|
distributed); transactional_components and optimistic_concurrency are declared
per-store so the consistency gate can cross-check each declared component
against a wired store."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.storage.features import (
    CoordinationScope,
    StorageComponent,
    TransactionScope,
)
from linktools.ai.runtime.persistence.features import StorageFeatures
from linktools.ai.runtime.persistence.facade import FilesystemStorage
from linktools.ai.runtime.persistence.sqlalchemy import (
    _ReferenceSqlAlchemyComposition,
)
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.storage.backends.sqlalchemy.models import Base as ObjectBase

_ALL = frozenset(StorageComponent)


def _sql_storage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/feat.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.run_sync(ObjectBase.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/feat.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _ReferenceSqlAlchemyComposition(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )


def test_file_storage_features_match_spec(tmp_path):
    # Features are DERIVED from the real wired objects on a FilesystemStorage,
    # not from caller-supplied constants.
    storage = FilesystemStorage(root=tmp_path)
    features = storage.features
    # transaction_scope=NONE: each file store is independently durable, but
    # there is NO general cross-store transaction (Storage.transaction()
    # raises). transactional_components is empty -- no components are grouped.
    # Features are DERIVED from real wired objects: only the ObjectStore
    # (assets) declares optimistic concurrency (versioning/CAS); the other
    # stores do not. append_only_events is False (the FS event store is not
    # declared append-only). This is the CORRECT derivation -- the old
    # hand-maintained FILE_STORAGE_FEATURES constant over-claimed.
    _assets_only = frozenset({StorageComponent.ASSETS})
    assert features == StorageFeatures(
        transaction_scope=TransactionScope.NONE,
        transactional_components=frozenset(),
        coordination_scope=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=_assets_only,
        append_only_events=False,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_artifacts=True,
        artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
    )


def test_sqlalchemy_storage_features_match_spec(tmp_path):
    storage = _sql_storage(tmp_path)
    features = storage.features
    # The in-repo SqlAlchemy reference ships the process-local coordinator,
    # so it declares PROCESS_LOCAL coordination. One AsyncSession groups every
    # store, so every component is transactional.
    # DERIVED: only ASSETS + ARTIFACT_RECORDS participate in the cross-store
    # UoW transaction (the other stores are not session-bound). The old
    # SQLALCHEMY_STORAGE_FEATURES constant claimed _ALL for both -- wrong.
    _txn = frozenset({StorageComponent.ASSETS, StorageComponent.ARTIFACT_RECORDS})
    assert features == StorageFeatures(
        transaction_scope=TransactionScope.DATABASE,
        transactional_components=_txn,
        coordination_scope=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=_txn,
        append_only_events=False,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_artifacts=True,
        artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
    )


def test_inrepo_references_do_not_claim_distributed_coordination(tmp_path):
    fs = FilesystemStorage(root=tmp_path).features
    sql = _sql_storage(tmp_path).features
    assert fs.coordination_scope is not CoordinationScope.DISTRIBUTED
    assert sql.coordination_scope is not CoordinationScope.DISTRIBUTED


def test_file_storage_cannot_do_database_transactions(tmp_path):
    features = FilesystemStorage(root=tmp_path).features
    assert features.transaction_scope is not TransactionScope.DATABASE


def test_is_frozen(tmp_path):
    features = FilesystemStorage(root=tmp_path).features
    with pytest.raises(Exception):
        features.transaction_scope = TransactionScope.DATABASE  # type: ignore[misc]
