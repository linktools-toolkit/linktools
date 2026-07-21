#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageFeatures: the capability surface each backend declares. Replaces the
former StorageCapabilities bool flags -- transaction/coordination are now
scopes (none/process_local/database|distributed) and streaming_blobs / fencing
are first-class so the capability gate can require them directly."""

from linktools.ai.storage.features import (
    FILE_STORAGE_FEATURES,
    SQLALCHEMY_STORAGE_FEATURES,
    CoordinationScope,
    StorageFeatures,
    TransactionScope,
)


def test_file_storage_features_match_spec():
    # transactions=NONE: each file store is independently durable, but there is
    # NO general cross-store transaction (Storage.transaction() raises). Claiming
    # PROCESS_LOCAL would over-state the capability (plan §4.5 Filesystem decision).
    assert FILE_STORAGE_FEATURES == StorageFeatures(
        transactions=TransactionScope.NONE,
        coordination=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=True,
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_blobs=True,
        full_text_search=False,
        semantic_search=False,
    )


def test_sqlalchemy_storage_features_match_spec():
    # The in-repo SqlAlchemyStorage reference ships the process-local
    # LocalLeaseCoordinator (no Redis/etcd in core), so it declares PROCESS_LOCAL
    # coordination. A downstream that injects a real distributed
    # LeaseCoordinator declares DISTRIBUTED on its own StorageFeatures; the
    # RuntimeBuilder gate then admits multi-worker topologies.
    assert SQLALCHEMY_STORAGE_FEATURES == StorageFeatures(
        transactions=TransactionScope.DATABASE,
        coordination=CoordinationScope.PROCESS_LOCAL,
        optimistic_concurrency=True,
        append_only_events=True,
        leasing=True,
        fencing=True,
        idempotency=True,
        streaming_blobs=True,
        full_text_search=True,
        semantic_search=False,
    )


def test_inrepo_references_do_not_claim_distributed_coordination():
    # The in-repo references (File + SQL) both ship process-local coordination.
    # Neither may claim DISTRIBUTED -- that is a downstream-injected capability,
    # and promising it without a distributed backend would be a fake capability.
    assert FILE_STORAGE_FEATURES.coordination is not CoordinationScope.DISTRIBUTED
    assert SQLALCHEMY_STORAGE_FEATURES.coordination is not CoordinationScope.DISTRIBUTED


def test_file_storage_cannot_do_database_transactions():
    # The whole point of the scope enum: FilesystemStorage is process-local, so the
    # builder's database-transaction branch (SqlAlchemyRunCommitCoordinator)
    # never fires for it.
    assert FILE_STORAGE_FEATURES.transactions is not TransactionScope.DATABASE


def test_is_frozen():
    import pytest

    with pytest.raises(Exception):
        FILE_STORAGE_FEATURES.transactions = TransactionScope.DATABASE  # type: ignore[misc]
