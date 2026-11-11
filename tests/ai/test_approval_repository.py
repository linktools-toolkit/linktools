#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval repository transaction reads and tenant visibility."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ApprovalStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ApprovalRecord
from linktools.ai.persistence.state import StateTransaction


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceStorage:
    if request.param == "memory":
        return PersistenceStorage.in_memory()
    return PersistenceStorage.sqlite(tmp_path / "approval.sqlite")


def _approval() -> ApprovalRecord:
    return ApprovalRecord(
        approval_id="approval", execution_id="execution", status=ApprovalStatus.PENDING,
        idempotency_key=None, decision=None, decided_by=None,
        decision_digest=None, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decided_at=None,
    )


@pytest.mark.asyncio
async def test_approval_transaction_getter_reads_new_record(storage: PersistenceStorage) -> None:
    async with Persistence.open("approval", storage=storage, tenant_id="tenant") as persistence:
        repository = persistence.checkpoint.approvals
        approval = _approval()

        async def create_and_read(transaction: StateTransaction) -> ApprovalRecord | None:
            await repository.create_in_transaction(transaction, approval)
            return await repository.get_approval_in_transaction(
                transaction, approval.approval_id, tenant_id="tenant",
            )

        assert await repository.state_store.mutate(create_and_read) == approval
        assert await repository.get(approval.approval_id, tenant_id="tenant") == approval


@pytest.mark.asyncio
async def test_approval_transaction_getter_returns_none_for_missing_record(storage: PersistenceStorage) -> None:
    async with Persistence.open("approval", storage=storage, tenant_id="tenant") as persistence:
        repository = persistence.checkpoint.approvals

        async def read(transaction: StateTransaction) -> ApprovalRecord | None:
            return await repository.get_approval_in_transaction(transaction, "missing", tenant_id="tenant")

        assert await repository.state_store.mutate(read) is None


@pytest.mark.asyncio
async def test_approval_transaction_getters_preserve_tenant_semantics(storage: PersistenceStorage) -> None:
    async with Persistence.open("approval", storage=storage, tenant_id="tenant") as persistence:
        repository = persistence.checkpoint.approvals
        approval = _approval()
        await repository.create(approval)

        async def read(transaction: StateTransaction) -> ApprovalRecord | None:
            assert await repository.get_approval_in_transaction(
                transaction, approval.approval_id, tenant_id="other",
            ) is None
            with pytest.raises(AIError) as error:
                await repository.get_in_transaction(transaction, approval.approval_id, tenant_id="other")
            assert error.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
            return await repository.get_approval_in_transaction(
                transaction, approval.approval_id, tenant_id="tenant",
            )

        assert await repository.state_store.mutate(read) == approval
        assert await repository.get(approval.approval_id, tenant_id="tenant") == approval
