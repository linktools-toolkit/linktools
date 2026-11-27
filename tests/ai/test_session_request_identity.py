#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session request identity stays independent from later mutable session state."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

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


@pytest.fixture
def storage() -> PersistenceStorage:
    return PersistenceStorage.in_memory()


@pytest.mark.parametrize("key", ("update", "é" * 64))
async def test_session_request_identity_update_replays_original_patch(storage: PersistenceStorage, key: str) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = UpdateSessionRequest(_PRINCIPAL, 0, key, {"first": True})
        first = await manager.update("agent", "session", request)
        receipt = await state.conversation.operations.get(key, tenant_id="tenant")
        assert receipt is not None and receipt.operation_id == key
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
        assert await state.conversation.operations.get("new", tenant_id="tenant") is None


async def test_session_request_identity_fork_replays_without_reinheriting(storage: PersistenceStorage) -> None:
    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = ForkSessionRequest(_PRINCIPAL, "child", "fork")
        first = await manager.fork("agent", "session", request)
        receipt = await state.conversation.operations.get("fork", tenant_id="tenant")
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
        receipt = await state.conversation.operations.get("shared", tenant_id="tenant")
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
        receipt = await state.conversation.operations.get("request", tenant_id="tenant")
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
        receipt = await state.conversation.operations.get("request", tenant_id="tenant")
        assert receipt is not None
        replay = await action("agent", "session", request)
        assert replay.session_id == ("session" if kind == "update" else "child")
        assert await state.conversation.operations.get(receipt.operation_id, tenant_id="tenant") == receipt


@pytest.mark.parametrize("kind", ("update", "fork"))
async def test_session_request_identity_reads_unmarked_receipt_without_rewriting(storage: PersistenceStorage, kind: str) -> None:
    import json
    from pathlib import Path
    from linktools.ai.persistence.state import operation_key

    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state, canonicalize_path=_canonicalize)
        await state.conversation.sessions.create(replace(_session(), cwd="original"))
        request = (UpdateSessionRequest(_PRINCIPAL, 0, "request", {}) if kind == "update"
                   else ForkSessionRequest(_PRINCIPAL, "child", "request"))
        action = manager.update if kind == "update" else manager.fork
        first = await action("agent", "session", request)
        data = json.loads((Path(__file__).parent / "fixtures/session_request_v1" / f"{kind}.json").read_text())
        key = operation_key("session-identity", "tenant", "conversation", "request")
        store = state.conversation.sessions.state_store

        async def install(transaction):
            stored = await transaction.get_operation(key)
            assert await transaction.replace_operation(replace(stored, data=data), expected_state=stored.state)

        await store.mutate(install)
        receipt = await state.conversation.operations.get("request", tenant_id="tenant")
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
        assert await state.conversation.operations.get("fork", tenant_id="tenant") is None


@pytest.mark.parametrize("helper", ("result", "failure", "status"))
async def test_session_request_identity_public_operation_helpers_preserve_contract(storage: PersistenceStorage, helper: str) -> None:
    from linktools.ai.core import OperationStatus
    from linktools.ai.execution import operation_failure, operation_result, operation_status

    async with Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state:
        manager = _manager(state)
        await state.conversation.sessions.create(_session())
        await manager.update("agent", "session", UpdateSessionRequest(_PRINCIPAL, 0, "request", {}))
        receipt = await state.conversation.operations.get("request", tenant_id="tenant")
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


@pytest.mark.parametrize("fault", ("unknown", "cancel"))
async def test_native_fork_replays_committed_request_after_source_and_child_change(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    import asyncio

    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    class HostAuthorization(TenantAuthorizationPolicy):
        pass

    storage = PersistenceStorage.sqlite(tmp_path / "native.db")
    options = {"principal": _PRINCIPAL, "expected_revision": 0, "idempotency_key": "fork"}
    async with (
        Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state,
        Runtime.open(persistence=state, features=RuntimeFeatures("native"), authorization=HostAuthorization()) as runtime,
    ):
        sessions = runtime.sessions
        await sessions.create_turn_session("session", principal=_PRINCIPAL, metadata={"initial": True})
        context = state.session_turns._context
        mutate = type(context).run_mutation

        async def commit_then_fault(self, operation, **kwargs):
            result = await mutate(self, operation, **kwargs)
            if self is not context or kwargs.get("domain") != "session.fork":
                return result
            assert result
            if fault == "cancel":
                raise asyncio.CancelledError()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(type(context), "run_mutation", commit_then_fault)
            with pytest.raises(asyncio.CancelledError if fault == "cancel" else AIError) as lost:
                await sessions.fork_turn_session("session", "child", **options)
            if fault == "unknown":
                assert lost.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN

    async with (
        Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state,
        Runtime.open(persistence=state, features=RuntimeFeatures("native"), authorization=HostAuthorization()) as runtime,
    ):
        sessions = runtime.sessions
        child = await state.session_turns.get_session("child")
        assert child["config_json"]["metadata"] == {"initial": True}
        assert await sessions.update_turn_session(
            "session", principal=_PRINCIPAL, expected_revision=1, metadata={"source": True},
        )
        assert await sessions.update_turn_session(
            "child", principal=_PRINCIPAL, expected_revision=0, metadata={"child": True},
        )
        assert not await sessions.update_turn_session(
            "child", principal=_PRINCIPAL, expected_revision=0, metadata={"stale": True},
        )
        changed = await state.session_turns.get_session("child")
        assert changed["config_json"]["metadata"] == {"child": True}
        assert changed["lineage_json"] == child["lineage_json"]
        source = await state.session_turns.get_session("session")
        assert await sessions.fork_turn_session("session", "child", **options)
        assert await state.session_turns.get_session("child") == changed
        assert await state.session_turns.get_session("session") == source
        with pytest.raises(AIError) as conflict:
            await sessions.fork_turn_session("session", "child", **(options | {"idempotency_key": "different"}))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT


async def test_native_fork_keeps_source_revision_fence(
    tmp_path: "Path", monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    class HostAuthorization(TenantAuthorizationPolicy):
        pass

    storage = PersistenceStorage.sqlite(tmp_path / "native.db")
    async with (
        Persistence.open("session-identity", storage=storage, tenant_id="tenant") as state,
        Runtime.open(persistence=state, features=RuntimeFeatures("native"), authorization=HostAuthorization()) as runtime,
    ):
        sessions = runtime.sessions
        await sessions.create_turn_session("session", principal=_PRINCIPAL)
        turns = state.session_turns
        original = turns.fork_session

        async def race(*args, **kwargs):
            assert await sessions.update_turn_session(
                "session", principal=_PRINCIPAL, expected_revision=0, metadata={"winner": True},
            )
            return await original(*args, **kwargs)

        monkeypatch.setattr(turns, "fork_session", race)
        assert not await sessions.fork_turn_session(
            "session", "child", principal=_PRINCIPAL, expected_revision=0, idempotency_key="fork",
        )
        assert await turns.get_session("child") is None
        source = await turns.get_session("session")
        assert source["revision"] == 1 and source["config_json"]["metadata"] == {"winner": True}
