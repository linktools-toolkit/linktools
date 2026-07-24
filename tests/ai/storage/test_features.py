#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageFeatures: the component-level capability surface each backend
declares. transaction/coordination are scopes (none/process_local/database|
distributed); transactional_components and optimistic_concurrency are declared
per-store so the consistency gate can cross-check each declared component
against a wired store."""

from linktools.ai.storage.features import (
    FILE_STORAGE_FEATURES,
    SQLALCHEMY_STORAGE_FEATURES,
    CoordinationScope,
    StorageComponent,
    StorageFeatures,
    TransactionScope,
)

_ALL = frozenset(StorageComponent)


def test_file_storage_features_match_spec():
    # transaction_scope=NONE: each file store is independently durable, but
    # there is NO general cross-store transaction (Storage.transaction()
    # raises). transactional_components is empty -- no components are grouped.
    assert FILE_STORAGE_FEATURES == StorageFeatures(
        transaction_scope=TransactionScope.NONE,
        transactional_components=frozenset(),
        coordination_scope=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=_ALL,
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_artifacts=True,
        artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
    )


def test_sqlalchemy_storage_features_match_spec():
    # The in-repo SqlAlchemyStorage reference ships the process-local
    # coordinator, so it declares PROCESS_LOCAL coordination. One AsyncSession
    # groups every store, so every component is transactional.
    assert SQLALCHEMY_STORAGE_FEATURES == StorageFeatures(
        transaction_scope=TransactionScope.DATABASE,
        transactional_components=_ALL,
        coordination_scope=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=_ALL,
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_artifacts=True,
        artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
    )


def test_inrepo_references_do_not_claim_distributed_coordination():
    # The in-repo references (File + SQL) both ship process-local coordination.
    assert (
        FILE_STORAGE_FEATURES.coordination_scope is not CoordinationScope.DISTRIBUTED
    )
    assert (
        SQLALCHEMY_STORAGE_FEATURES.coordination_scope
        is not CoordinationScope.DISTRIBUTED
    )


def test_file_storage_cannot_do_database_transactions():
    # The whole point of the scope enum: FilesystemStorage is process-local, so
    # the builder's database-transaction branch never fires for it.
    assert FILE_STORAGE_FEATURES.transaction_scope is not TransactionScope.DATABASE


def test_is_frozen():
    import pytest

    with pytest.raises(Exception):
        FILE_STORAGE_FEATURES.transaction_scope = TransactionScope.DATABASE  # type: ignore[misc]
