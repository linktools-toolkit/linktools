#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session operation replay preserves ownership and revision fencing."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import OperationLedgerInput
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.session import CreateSessionRequest, UpdateSessionRequest

from .test_session_metadata_mapping import _PRINCIPAL, _manager

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceStorage:
    return (PersistenceStorage.in_memory() if request.param == "memory"
            else PersistenceStorage.sqlite(tmp_path / "state.db"))


async def test_session_replay_checks_owner_before_receipt(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-replay", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await manager.create("agent", CreateSessionRequest(_PRINCIPAL, "session", "create"))
        await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "update", {}))
        record = await state.conversation.sessions.get("session", tenant_id="tenant")
        receipt = await state.conversation.operations.get("update", tenant_id="tenant")
        assert record is not None and receipt is not None
        operation = OperationLedgerInput(
            receipt.operation_id, receipt.tenant_id, receipt.resource_kind,
            receipt.resource_id, receipt.execution_id, receipt.operation_kind,
            receipt.status, receipt.request_digest, receipt.result_ref,
            receipt.result_digest, receipt.error_code, receipt.compactable,
            receipt.created_at, receipt.updated_at,
        )
        with pytest.raises(AIError) as denied:
            await state.conversation.sessions.compare_and_swap_with_operation(
                "session", tenant_id="tenant", expected_revision=0,
                next_record=replace(record, owner_principal_id="other"), operation=operation,
            )
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await state.conversation.sessions.get("session", tenant_id="tenant") == record
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt


async def test_session_update_revision_race_rolls_back_losing_receipt(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-replay", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await manager.create("agent", CreateSessionRequest(_PRINCIPAL, "session", "create"))
        requests = [UpdateSessionRequest(_PRINCIPAL, 0, key, {"key": key}) for key in ("first", "second")]
        results = await asyncio.gather(
            *(manager.update("agent", "session", request) for request in requests),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, BaseException) for result in results) == 1
        for request, result in zip(requests, results):
            receipt = await state.conversation.operations.get(
                request.idempotency_key, tenant_id="tenant")
            if isinstance(result, BaseException):
                assert isinstance(result, AIError)
                assert result.code is ErrorCode.SESSION_REVISION_CONFLICT
                assert receipt is None
            else:
                assert receipt is not None
                assert result.revision == 1
                assert await manager.update("agent", "session", request) == result
                assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        record = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert record is not None and record.revision == 1
