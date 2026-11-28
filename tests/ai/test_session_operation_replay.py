#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session operation replay preserves ownership and revision fencing."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, OperationLedgerInput, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import CreateSessionRequest, UpdateSessionRequest
from pydantic_ai.messages import ModelResponse, TextPart

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

from .test_session_metadata_mapping import _PRINCIPAL, _manager

pytestmark = pytest.mark.asyncio


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceStorage:
    return (PersistenceStorage.in_memory() if request.param == "memory"
            else PersistenceStorage.sqlite(tmp_path / "state.db"))


async def test_memory_session_replay_checks_owner_before_receipt() -> None:
    async with Persistence.open("session-replay", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
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


async def test_memory_session_update_revision_race_rolls_back_losing_receipt() -> None:
    async with Persistence.open("session-replay", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
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

class _HostSessionAuthority(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)
        if principal.principal_id != "owner":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def _features(model):
    features = RuntimeFeatures("native-session-replay")
    features.agent("default", model="default", allow_tools=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


async def test_native_session_replay_obeys_host_session_authority(tmp_path: Path, monkeypatch) -> None:
    calls = []

    async def model(_messages, _info):
        calls.append(True)
        return ModelResponse(parts=[TextPart("answer")])

    async with Runtime.open(
        "native-session-replay", storage=PersistenceStorage.from_root(tmp_path), tenant_id="tenant",
        features=_features(model), authorization=_HostSessionAuthority(),
    ) as runtime:
        owner = Principal(principal_id="owner", tenant_id="tenant")
        outsider = Principal(principal_id="outsider", tenant_id="tenant")
        await runtime.sessions.create_turn_session("session", principal=owner)
        async def start(actor):
            return await runtime.sessions.start_turn(
                "session", "question", principal=actor, agent="default", idempotency_key="request",
            )

        original = await start(owner)
        assert (await original.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        turns = runtime._state.session_turns
        receipt = await turns.find_request("session", "request")

        async def forbidden_read(*_args, **_kwargs):
            pytest.fail("denied session access must not read the turn receipt")

        with monkeypatch.context() as patch:
            patch.setattr(turns, "find_request", forbidden_read)
            with pytest.raises(AIError) as denied:
                await start(outsider)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        replay = await start(owner)
        assert replay.key == original.key and calls == [True]
        assert await runtime._state.session_turns.find_request("session", "request") == receipt


async def test_native_session_update_cas_has_one_winner_and_no_shadow_receipt(tmp_path: Path) -> None:
    async def model(_messages, _info):
        pytest.fail("Session metadata never dispatches provider work")

    storage = PersistenceStorage.from_root(tmp_path)
    features = _features(model)
    owner = Principal(principal_id="owner", tenant_id="tenant")
    async with (
        Runtime.open("native-session-replay", storage=storage, tenant_id="tenant",
                     features=features, authorization=_HostSessionAuthority()) as first,
        Runtime.open("native-session-replay", storage=storage, tenant_id="tenant",
                     features=features, authorization=_HostSessionAuthority()) as second,
    ):
        await first.sessions.create_turn_session("session", principal=owner)
        assert first._state.session_turns._context.engine is not second._state.session_turns._context.engine

        async def update(host, value):
            return await host.sessions.update_turn_session(
                "session", principal=owner, expected_revision=0, metadata={"key": value},
            )

        results = await asyncio.gather(update(first, "first"), update(second, "second"))
        assert results.count(True) == results.count(False) == 1
        winning, losing = ("first", "second") if results[0] else ("second", "first")
        assert not await update(first, losing)
        for host in (first, second):
            state = host._state
            assert state.conversation.sessions is None
            row = await state.session_turns.get_session("session")
            assert row["revision"] == 1 and row["config_json"]["metadata"] == {"key": winning}
            facts = await state.conversation.histories.state_store.read(lambda tx: tx.scan_records())
            assert not any(record.kind in {"session", "session_turn_commit"} for record in facts)
