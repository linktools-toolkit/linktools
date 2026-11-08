#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session request identity stays independent from later mutable session state."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from linktools.ai.core import idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.session import ForkSessionRequest, UpdateSessionRequest

from .test_session_management import _PRINCIPAL, _manager
from .test_session_timeline import _session

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


async def _canonicalize(value):
    return value


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: "Path") -> PersistenceStorage:
    return (PersistenceStorage.in_memory() if request.param == "memory"
            else PersistenceStorage.sqlite(tmp_path / "state.db"))


async def test_session_request_identity_update_replays_original_patch(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = UpdateSessionRequest(_PRINCIPAL, 0, "update", {"first": True})
        first = await manager.update("agent", "session", request)
        receipt = await state.conversation.operations.get(idempotency_key_digest("update"), tenant_id="tenant")
        changed = await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, first.revision, "later", {"later": True}, "changed"))
        assert await manager.update("agent", "session", request) == changed
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        with pytest.raises(AIError) as conflict:
            await manager.update("agent", "session", replace(request, cwd="changed"))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as revision:
            await manager.update("agent", "session", replace(request, idempotency_key="new"))
        assert revision.value.code is ErrorCode.SESSION_REVISION_CONFLICT
        assert await state.conversation.operations.get(idempotency_key_digest("new"), tenant_id="tenant") is None


async def test_session_request_identity_fork_replays_without_reinheriting(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = ForkSessionRequest(_PRINCIPAL, "child", "fork")
        first = await manager.fork("agent", "session", request)
        receipt = await state.conversation.operations.get(idempotency_key_digest("fork"), tenant_id="tenant")
        await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, 0, "source-change", {"source": True}, "source-new"))
        changed = await manager.update("agent", "child", UpdateSessionRequest(
            _PRINCIPAL, first.revision, "child-change", {"child": True}, "child-new"))
        assert await manager.fork("agent", "session", request) == changed
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        with pytest.raises(AIError) as conflict:
            await manager.fork("agent", "session", replace(request, cwd="source-new"))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.parametrize("scenario", ("update-fork", "fork-update", "update-target", "fork-target"))
async def test_session_request_identity_dispatches_receipt_owner_before_contract(storage: PersistenceStorage, scenario: str) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(_session())
        await state.conversation.sessions.create(replace(_session(), session_id="other"))
        if scenario.startswith("update"):
            await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "shared", {}))
        else:
            await manager.fork("agent", "session", ForkSessionRequest(_PRINCIPAL, "child", "shared"))
        receipt = await state.conversation.operations.get(idempotency_key_digest("shared"), tenant_id="tenant")
        with pytest.raises(AIError) as conflict:
            if scenario == "update-fork":
                await manager.fork("agent", "session", ForkSessionRequest(_PRINCIPAL, "child", "shared"))
            elif scenario == "fork-target":
                await manager.fork("agent", "session", ForkSessionRequest(_PRINCIPAL, "other-child", "shared"))
            else:
                target = "other" if scenario == "update-target" else "session"
                await manager.update("agent", target, UpdateSessionRequest(_PRINCIPAL, 0, "shared", {}))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt


@pytest.mark.parametrize("kind", ("update", "fork"))
async def test_session_request_identity_marker_cannot_change_under_cas(storage: PersistenceStorage, kind: str) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        if kind == "update":
            await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "request", {}))
        else:
            await manager.fork("agent", "session", ForkSessionRequest(_PRINCIPAL, "child", "request"))
        receipt = await state.conversation.operations.get(idempotency_key_digest("request"), tenant_id="tenant")
        with pytest.raises(AIError) as conflict:
            await state.conversation.operations.compare_and_swap(
                receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
                next_record=replace(receipt, request_digest_contract=None))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt


@pytest.mark.parametrize("kind", ("update", "fork"))
async def test_session_request_identity_rejects_unknown_owned_contract(storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = (UpdateSessionRequest(_PRINCIPAL, 0, "request", {}) if kind == "update"
                   else ForkSessionRequest(_PRINCIPAL, "child", "request"))
        action = manager.update if kind == "update" else manager.fork
        await action("agent", "session", request)
        original = state.conversation.operations.get

        async def unknown(*args, **kwargs):
            receipt = await original(*args, **kwargs)
            return None if receipt is None else replace(receipt, request_digest_contract="unknown")

        monkeypatch.setattr(state.conversation.operations, "get", unknown)
        with pytest.raises(AIError) as corruption:
            await action("agent", "session", request)
        assert corruption.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("kind", ("update", "fork"))
@pytest.mark.parametrize("fault", ("unknown", "cancel"))
async def test_session_request_identity_retry_after_committed_response_fault(storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch, kind: str, fault: str) -> None:
    import asyncio

    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        request = (UpdateSessionRequest(_PRINCIPAL, 0, "request", {}) if kind == "update"
                   else ForkSessionRequest(_PRINCIPAL, "child", "request"))
        action = manager.update if kind == "update" else manager.fork
        store = state.conversation.sessions.state_store
        original = store.mutate
        raised = False

        async def commit_then_fault(operation):
            nonlocal raised
            result = await original(operation)
            if not raised:
                raised = True
                if fault == "cancel":
                    raise asyncio.CancelledError()
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result

        monkeypatch.setattr(store, "mutate", commit_then_fault)
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else AIError):
            await action("agent", "session", request)
        monkeypatch.setattr(store, "mutate", original)
        receipt = await state.conversation.operations.get(idempotency_key_digest("request"), tenant_id="tenant")
        assert receipt is not None
        replay = await action("agent", "session", request)
        assert replay.session_id == ("session" if kind == "update" else "child")
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt


@pytest.mark.parametrize("kind", ("update", "fork"))
async def test_session_request_identity_reads_real_legacy_receipt_without_rewriting(storage: PersistenceStorage, kind: str) -> None:
    import json
    from pathlib import Path
    from linktools.ai.core import PersistenceDomain
    from linktools.ai.persistence.state import operation_key

    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = (UpdateSessionRequest(_PRINCIPAL, 0, "request", {}) if kind == "update"
                   else ForkSessionRequest(_PRINCIPAL, "child", "request"))
        action = manager.update if kind == "update" else manager.fork
        first = await action("agent", "session", request)
        data = json.loads((Path(__file__).parent / "fixtures/session_request_v1" / f"{kind}.json").read_text())
        key = operation_key("session-identity", "tenant", PersistenceDomain.CONVERSATION.value,
                            idempotency_key_digest("request"))
        store = state.conversation.sessions.state_store

        async def install(transaction):
            stored = await transaction.get_operation(key)
            assert await transaction.replace_operation(replace(stored, data=data), expected_state=stored.state)

        await store.mutate(install)
        receipt = await state.conversation.operations.get(idempotency_key_digest("request"), tenant_id="tenant")
        assert receipt.request_digest_contract is None
        assert await action("agent", "session", request) == first
        current = await state.conversation.sessions.get("session", tenant_id="tenant")
        await manager.update("agent", "session", UpdateSessionRequest(
            _PRINCIPAL, current.revision, "change", {"later": True}, "changed"))
        with pytest.raises(AIError) as conflict:
            await action("agent", "session", request)
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        result = await action("agent", "session", replace(request, cwd="original"))
        assert result.session_id == first.session_id
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt
        stored = await store.read(lambda transaction: transaction.get_operation(key))
        assert stored.data == data


async def test_session_request_identity_fork_keeps_source_revision_fence(storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        original = state.conversation.sessions.create_fork_with_operation

        async def race(*args, **kwargs):
            await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "race", {}))
            return await original(*args, **kwargs)

        monkeypatch.setattr(state.conversation.sessions, "create_fork_with_operation", race)
        with pytest.raises(AIError) as conflict:
            await manager.fork("agent", "session", ForkSessionRequest(_PRINCIPAL, "child", "fork"))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        assert await state.conversation.sessions.get("child", tenant_id="tenant") is None
        assert await state.conversation.operations.get(idempotency_key_digest("fork"), tenant_id="tenant") is None


@pytest.mark.parametrize("helper", ("result", "failure", "status"))
async def test_session_request_identity_public_operation_helpers_preserve_contract(storage: PersistenceStorage, helper: str) -> None:
    from linktools.ai.core import OperationStatus
    from linktools.ai.execution import operation_failure, operation_result, operation_status

    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "request", {}))
        receipt = await state.conversation.operations.get(idempotency_key_digest("request"), tenant_id="tenant")
        if helper == "result":
            candidate = operation_result(receipt, receipt.result_ref, receipt.result_digest)
        elif helper == "failure":
            candidate = operation_failure(receipt, "failure")
        else:
            candidate = operation_status(receipt, receipt.status)
        assert candidate.request_digest_contract == receipt.request_digest_contract == "session-update-request-v2"
        assert replace(candidate, status=receipt.status, result_ref=receipt.result_ref,
                       result_digest=receipt.result_digest, error_code=receipt.error_code,
                       updated_at=receipt.updated_at) == receipt
        if helper == "failure":
            assert candidate.status is OperationStatus.FAILED
            with pytest.raises(AIError) as conflict:
                await state.conversation.operations.compare_and_swap(
                    receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
                    next_record=candidate)
            assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        else:
            assert await state.conversation.operations.compare_and_swap(
                receipt.operation_id, tenant_id="tenant", expected_status=receipt.status,
                next_record=candidate) == candidate
