#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close ownership, version fencing and receipt contracts by storage owner."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.core import OperationStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import CloseSessionRequest, UpdateSessionRequest

from .test_session_management import _PRINCIPAL, _manager
from .test_session_timeline import _session
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

pytestmark = pytest.mark.asyncio

async def test_session_request_identity_close_receipt_survives_closed_update() -> None:
    async with Persistence.open("session-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = CloseSessionRequest(_PRINCIPAL, "close")
        closed = await manager.close("session", request)
        receipt = await state.conversation.operations.get("close", tenant_id="tenant")
        updated = await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, closed.revision, "update-closed", {"legal": True}))
        assert await manager.close("session", request) == updated
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        with pytest.raises(AIError) as changed:
            await manager.close("session", replace(request, force=True))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert await manager.get("session", principal=_PRINCIPAL) == updated


@pytest.mark.parametrize("digest", (None, "bad", "A" * 64))
async def test_session_request_identity_close_rejects_malformed_receipt(digest: str | None) -> None:
    async with Persistence.open("session-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = CloseSessionRequest(_PRINCIPAL, "close")
        await manager.close("session", request)
        receipt = await state.conversation.operations.get("close", tenant_id="tenant")
        await state.conversation.operations.compare_and_swap(
            receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
            next_record=replace(receipt, result_digest=digest))
        with pytest.raises(AIError) as corruption:
            await manager.close("session", request)
        assert corruption.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("status", (OperationStatus.PENDING, OperationStatus.SUCCEEDED))
async def test_session_close_rejects_unknown_receipt_contract(
    monkeypatch: pytest.MonkeyPatch, status: OperationStatus,
) -> None:
    async with Persistence.open("session-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("session-identity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
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

class _SessionAuthority(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)
        if principal.principal_id != "owner":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


async def test_native_close_authorizes_before_cas_and_cannot_cross_a_revision(tmp_path, monkeypatch):
    async def model(_messages, _info):
        pytest.fail("Closing a session must not dispatch a model")

    owner = Principal(principal_id="owner", tenant_id="tenant")
    outsider = Principal(principal_id="outsider", tenant_id="tenant")
    storage = PersistenceStorage.from_root(tmp_path)
    features = RuntimeFeatures("native-session-close")
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(_UsageFunctionModel(model)), features)

    async with (
        Runtime.open("native-session-close", storage=storage, tenant_id="tenant",
                     authorization=_SessionAuthority(), features=features) as first,
        Runtime.open("native-session-close", storage=storage, tenant_id="tenant",
                     authorization=_SessionAuthority(), features=features) as second,
    ):
        await first.sessions.create_turn_session("session", principal=owner)
        turns = first._state.session_turns
        assert turns._context.engine is not second._state.session_turns._context.engine
        initial = await turns.get_session("session")

        async def forbidden_write(*_args, **_kwargs):
            pytest.fail("Denied principal reached the native close CAS")

        with monkeypatch.context() as patch:
            patch.setattr(turns, "close_session", forbidden_write)
            with pytest.raises(AIError) as rejected:
                await first.sessions.close_turn_session("session", principal=outsider, expected_revision=0)
            assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await turns.get_session("session") == initial
        assert not await first.sessions.close_turn_session("session", principal=owner, expected_revision=1)

        closed, updated = await asyncio.gather(
            first.sessions.close_turn_session("session", principal=owner, expected_revision=0),
            second.sessions.update_turn_session("session", principal=owner, expected_revision=0,
                                               metadata={"winner": "update"}),
        )
        assert closed is not updated
        after_race = await turns.get_session("session")
        assert after_race["revision"] == 1
        if updated:
            assert after_race["status"] == "open" and after_race["config_json"]["metadata"] == {"winner": "update"}
            assert await first.sessions.close_turn_session("session", principal=owner, expected_revision=1)
        else:
            assert after_race["status"] == "closed" and after_race["config_json"] == initial["config_json"]

        current = await turns.get_session("session")
        assert current["status"] == "closed"
        assert not await second.sessions.close_turn_session("session", principal=owner, expected_revision=0)
        assert not await second.sessions.update_turn_session("session", principal=owner,
                                                             expected_revision=current["revision"], metadata={"stale": True})
        assert await second._state.session_turns.get_session("session") == current
        assert first._state.conversation.sessions is None
        records = await first._state.conversation.histories.state_store.read(lambda tx: tx.scan_records())
        assert not any(record.kind in {"session", "session_turn_commit"} for record in records)
