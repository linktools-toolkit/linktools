#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed close receipts remain valid after permitted session metadata updates."""

from dataclasses import replace

import pytest

from linktools.ai.core import OperationStatus, idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.session import CloseSessionRequest, UpdateSessionRequest

from .test_session_management import _PRINCIPAL, _manager
from .test_session_timeline import _session
from .test_session_operation_replay import storage

pytestmark = pytest.mark.asyncio

async def test_session_request_identity_close_receipt_survives_closed_update(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = CloseSessionRequest(_PRINCIPAL, "close")
        closed = await manager.close("session", request)
        receipt = await state.conversation.operations.get(idempotency_key_digest("close"), tenant_id="tenant")
        updated = await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, closed.revision, "update-closed", {"legal": True}))
        assert await manager.close("session", request) == updated
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        with pytest.raises(AIError) as changed:
            await manager.close("session", replace(request, force=True))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await manager.get("session", principal=_PRINCIPAL) == updated


@pytest.mark.parametrize("digest", (None, "bad", "A" * 64))
async def test_session_request_identity_close_rejects_malformed_receipt(storage: PersistenceStorage, digest: str | None) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = CloseSessionRequest(_PRINCIPAL, "close")
        await manager.close("session", request)
        receipt = await state.conversation.operations.get(idempotency_key_digest("close"), tenant_id="tenant")
        await state.conversation.operations.compare_and_swap(
            receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
            next_record=replace(receipt, result_digest=digest))
        with pytest.raises(AIError) as corruption:
            await manager.close("session", request)
        assert corruption.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("status", (OperationStatus.PENDING, OperationStatus.SUCCEEDED))
async def test_session_close_rejects_unknown_receipt_contract(
    storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch, status: OperationStatus,
) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = CloseSessionRequest(_PRINCIPAL, "close")
        closed = await manager.close("session", request)
        original = state.conversation.operations.get

        async def unknown(*args, **kwargs):
            receipt = await original(*args, **kwargs)
            if receipt is None:
                return None
            return replace(
                receipt, request_digest_contract="unsupported-close-contract", status=status,
                result_ref=receipt.result_ref if status is OperationStatus.SUCCEEDED else None,
                result_digest=receipt.result_digest if status is OperationStatus.SUCCEEDED else None,
            )

        monkeypatch.setattr(state.conversation.operations, "get", unknown)
        with pytest.raises(AIError) as corruption:
            await manager.close("session", request)
        assert corruption.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await manager.get("session", principal=_PRINCIPAL) == closed


async def test_session_close_completion_checks_racing_receipt_digest(
    storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        original = state.conversation.operations.compare_and_swap

        async def conflicting_completion(operation_id, **kwargs):
            candidate = kwargs["next_record"]
            if candidate.status is OperationStatus.SUCCEEDED:
                await original(operation_id, **{**kwargs, "next_record": replace(candidate, result_digest="a" * 64)})
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await original(operation_id, **kwargs)

        monkeypatch.setattr(state.conversation.operations, "compare_and_swap", conflicting_completion)
        with pytest.raises(AIError) as corruption:
            await manager.close("session", CloseSessionRequest(_PRINCIPAL, "close"))
        assert corruption.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
