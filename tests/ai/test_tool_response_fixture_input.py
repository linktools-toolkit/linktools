#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool response fixtures belong to accepted invocation input and durable lineage."""

import base64
import json
import sqlite3
import zlib
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path

import pytest

from linktools.ai.asset import AssetKey, AssetStore, AssetVersionRef, InMemoryAssetBackend
from linktools.ai.core import AuthorizationAction, ExecutionLineageKind, ExecutionStatus, Principal, ResourceRef, canonical_sha256, principal_identity_payload
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CaptureInputRequest, ExecutionRequest, Runtime, RuntimeStorage, ToolResponseFixture
from linktools.ai.runtime._execution import _request_digest
from linktools.ai.runtime._input import input_intent
from linktools.ai.runtime.state._codec import _decode_enveloped_domain, decode_domain, encode_domain
from linktools.ai.runtime.state._contracts import ExecutionRecord, ExecutionStartReservation
from linktools.ai.storage import StorageEntryRevision, StorageOverlay

from ._runtime_test_helpers import RuntimeUsageModels


def _reference(name: str = "case-a") -> AssetVersionRef:
    return AssetVersionRef(AssetKey("tool-response", name), "fixture-layer", StorageEntryRevision(1), "a" * 64, 42)


def _historical_execution() -> ExecutionRecord:
    path = Path(__file__).parent / "fixtures" / "persistence" / "sqlite_session_budget_4609177e.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    sql = zlib.decompress(base64.b64decode(fixture["sql_zlib_base64"]))
    with sqlite3.connect(":memory:") as connection:
        connection.executescript(sql.decode("utf-8"))
        row = connection.execute("SELECT payload_json FROM ai_state_records WHERE kind = 'execution'").fetchone()
    payload = json.loads(row[0])
    assert "tool_response_ref" not in payload["value"]["payload"]["fields"]
    return _decode_enveloped_domain(payload, ExecutionRecord)


def test_fixture_reference_is_optional_in_historical_execution_wire() -> None:
    historical = _historical_execution()
    assert historical.tool_response_ref is None
    for reference in (None, _reference()):
        current = replace(historical, tool_response_ref=reference)
        assert decode_domain(encode_domain(current), ExecutionRecord) == current
    wire = encode_domain(historical)
    wire["fields"].pop("tool_response_ref")
    assert decode_domain(wire, ExecutionRecord) == historical
    wire["fields"]["tool_response_ref"] = "not-an-asset-version"
    with pytest.raises(AIError) as invalid:
        decode_domain(wire, ExecutionRecord)
    assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_fixture_request_identity_preserves_live_digest_and_pins_asset_version() -> None:
    request = ExecutionRequest("question", Principal("owner", "tenant"), "request-key", None, "run", False, False)
    lineage = dict(session_id=None, previous_execution_id=None, fork_base_execution_id=None,
                   parent_execution_id=None, root_execution_id=None, parent_invocation_id=None,
                   lineage_kind=ExecutionLineageKind.RUN)
    original = canonical_sha256({
        "input_intent": input_intent("question", ()).digest,
        "binding_digest": "b" * 64, "scope": "execution",
        "principal": principal_identity_payload(request.principal),
        "session_id": None, "previous_execution_id": None, "fork_base_execution_id": None,
        "parent_execution_id": None, "parent_invocation_id": None, "root_identity": "$self",
        "lineage_kind": ExecutionLineageKind.RUN.value, "memory_scope_digest": None,
        "mode": "run", "planning": False, "thinking": False,
    })
    assert _request_digest(request, "b" * 64, **lineage) == original
    first = replace(request, tool_response_ref=_reference())
    second = replace(request, tool_response_ref=_reference("case-b"))
    assert len({_request_digest(value, "b" * 64, **lineage) for value in (request, first, second)}) == 3
    with pytest.raises(AIError) as invalid:
        replace(request, tool_response_ref="not-an-asset-version")
    assert invalid.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def _fixtures() -> tuple[AssetStore, AssetVersionRef, AssetVersionRef]:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    keys = (AssetKey("tool-response", "case-a"), AssetKey("tool-response", "case-b"))
    for key in keys:
        await store.put(key, b'{"version":1,"kind":"mcp-tool-responses","servers":[]}')
    first, second = await store.resolve_versions(keys)
    return store, first, second


@pytest.mark.asyncio
async def test_capture_retry_fork_and_child_keep_saved_fixture_after_default_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, first, second = await _fixtures()
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("fixture-input", models=RuntimeUsageModels(), storage=storage,
                            tool_responses=ToolResponseFixture(first, assets)) as runtime:
        execution = await runtime.agents.get().start("first", idempotency_key="first")
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record.tool_response_ref == first
        assert record.context_imported is False
        principal = runtime.default_principal
        for policy in ("clean", "captured"):
            reference = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(principal, "capture-" + policy, policy))
            capture = await runtime._input_captures.read_agent(reference, principal=principal)
            assert capture.tool_response_ref == first
            assert (capture.input_context is None) == (policy == "clean")
            with pytest.raises(AIError) as unsupported:
                await runtime._input_captures.resolve_task_input(reference, principal=principal)
            assert unsupported.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
            assert unsupported.value.safe_details["reason"] == "tool_response_fixture_import_unsupported"
            with pytest.raises(AIError) as task_import:
                await runtime.tasks.from_agent_capture("capture.task", reference, principal=principal)
            assert task_import.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        async def no_asset_read(refs: Sequence[AssetVersionRef]) -> tuple[bytes, ...]:
            raise AssertionError("a completed idempotent replay does not read fixture Assets")

        with monkeypatch.context() as patch:
            patch.setattr(assets, "read_versions", no_asset_read)
            replay = await runtime.agents.get().start("first", idempotency_key="first")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait()).result.status is ExecutionStatus.SUCCEEDED
    assert await assets.read_versions((first,))

    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("fixture-input", models=RuntimeUsageModels(), storage=storage,
                            tool_responses=ToolResponseFixture(second, assets)) as runtime:
        original = await runtime.executions.get(execution.execution_id)
        for derived in (await original.retry("retry"), await original.fork("fork")):
            assert (await derived.wait()).result.status is ExecutionStatus.SUCCEEDED
            accepted = await storage.execution.executions.get(derived.execution_id, tenant_id=runtime.tenant_id)
            assert accepted.tool_response_ref == first
        request = ExecutionRequest("child", runtime.default_principal, "child", None, "run", False, False,
                                   tool_response_ref=second)
        handle = await runtime._execution_service.start_subagent(record.binding_digest, request,
            parent_execution_id=record.execution_id, root_execution_id=record.root_execution_id,
            parent_invocation_id="child-call", binding_contract=record.binding)
        child = await runtime.executions.get(handle.execution_id)
        assert (await child.wait()).result.status is ExecutionStatus.SUCCEEDED
        accepted = await storage.execution.executions.get(child.execution_id, tenant_id=runtime.tenant_id)
        assert accepted.tool_response_ref == first
        assert accepted.context_imported is False
        restored = await runtime._execution_service._request_for_execution(request, record)
        assert restored.tool_response_ref == first
        legacy = await runtime._execution_service._request_for_execution(request, replace(record, tool_response_ref=None))
        assert legacy.tool_response_ref is None
        with pytest.raises(AIError) as conflict:
            await runtime.agents.get().start("first", idempotency_key="first")
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    await assets.close()


@pytest.mark.asyncio
async def test_fixture_session_rejection_leaves_no_admitted_turn(tmp_path: Path) -> None:
    assets, first, _second = await _fixtures()
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("fixture-session", models=RuntimeUsageModels(), storage=storage,
                            tool_responses=ToolResponseFixture(first, assets)) as runtime:
        with pytest.raises(AIError) as rejected:
            await runtime.agents.get().start("question", session_id="session")
        assert rejected.value.code is ErrorCode.BINDING_CONFLICT
        assert rejected.value.safe_details["reason"] == "tool_response_fixture_session_unsupported"
        session = await storage.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
        assert session.active_execution_id is None
        assert await storage.conversation.sessions.timeline_head("session", tenant_id=runtime.tenant_id) == 0
        assert await storage.execution.executions.list_by_session("session", tenant_id=runtime.tenant_id) == ()
    await assets.close()


@pytest.mark.asyncio
async def test_derived_fixture_mode_mismatch_fails_before_execution_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assets, first, _second = await _fixtures()
    async with Runtime.open("fixture-modes", models=RuntimeUsageModels(), storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        live = await runtime.agents.get().start("live")
        assert (await live.wait()).result.status is ExecutionStatus.SUCCEEDED
    async with Runtime.open("fixture-modes", models=RuntimeUsageModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            tool_responses=ToolResponseFixture(first, assets)) as runtime:
        fixture = await runtime.agents.get().start("fixture")
        assert (await fixture.wait()).result.status is ExecutionStatus.SUCCEEDED
        original = await runtime.executions.get(live.execution_id)
        with pytest.raises(AIError) as mismatch:
            await original.retry("retain live mode")
        assert mismatch.value.code is ErrorCode.BINDING_CONFLICT
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("fixture-modes", models=RuntimeUsageModels(), storage=storage) as runtime:
        async def unexpected_reservation(value: ExecutionStartReservation) -> None:
            raise AssertionError("fixture mismatch must fail before execution admission")

        monkeypatch.setattr(storage.execution.executions, "reserve_start", unexpected_reservation)
        original = await runtime.executions.get(fixture.execution_id)
        with pytest.raises(AIError) as missing:
            await original.fork("retain fixture mode")
        assert missing.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
    await assets.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", (False, True))
async def test_fixture_asset_preflight_is_authorized_and_precedes_admission(authorized: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    assets, first, _second = await _fixtures()
    reads = []

    class Authorization:
        async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
            if action is AuthorizationAction.EXECUTION_RUN and not authorized:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def missing_version(refs: Sequence[AssetVersionRef]) -> tuple[bytes, ...]:
        reads.extend(refs)
        raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)

    async def unexpected_reservation(value: ExecutionStartReservation) -> None:
        raise AssertionError("fixture preflight must precede execution admission")

    monkeypatch.setattr(assets, "read_versions", missing_version)
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("fixture-preflight", models=RuntimeUsageModels(), storage=storage,
                            authorization=Authorization(), tool_responses=ToolResponseFixture(first, assets)) as runtime:
        monkeypatch.setattr(storage.execution.executions, "reserve_start", unexpected_reservation)
        with pytest.raises(AIError) as rejected:
            await runtime.agents.get().start("question")
        assert rejected.value.code is (ErrorCode.ASSET_VERSION_NOT_FOUND if authorized else ErrorCode.AUTHORIZATION_DENIED)
        assert reads == ([first] if authorized else [])
    await assets.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", (
    "pending_admission", "admitted_first_attempt", "active_startup", "explicit_recovery",
    "ready_deferred", "pending_deferred", "cancel_recovery", "terminal_cleanup",
))
async def test_recovery_reads_fixture_only_before_actual_resume_and_preserves_original_on_failure(path: str) -> None:
    from types import SimpleNamespace

    from linktools.ai.core import ExternalCallStatus
    from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
    from linktools.ai.runtime.state._contracts import (
        PendingDeferredCall, PendingToolContinuation, RecoveryCheckpoint, RecoveryCheckpointState,
    )
    from linktools.ai.storage import StoredPayload

    historical = _historical_execution()
    status = {
        "pending_admission": ExecutionStatus.PENDING_START,
        "explicit_recovery": ExecutionStatus.RECOVERY_REQUIRED,
        "ready_deferred": ExecutionStatus.WAITING_DEFERRED,
        "pending_deferred": ExecutionStatus.WAITING_DEFERRED,
        "cancel_recovery": ExecutionStatus.CANCELLING,
        "terminal_cleanup": ExecutionStatus.SUCCEEDED,
    }.get(path, ExecutionStatus.STARTED)
    state = (RecoveryCheckpointState.ADMITTED if path in {"pending_admission", "admitted_first_attempt"}
             else RecoveryCheckpointState.WAITING if path in {"ready_deferred", "pending_deferred"}
             else RecoveryCheckpointState.ACTIVE)
    execution = replace(historical, session_id=None, status=status, tool_response_ref=_reference())
    pending = (PendingToolContinuation("run", calls=(
        PendingDeferredCall("call", "lookup", StoredPayload.inline_json({})),
    )) if state is RecoveryCheckpointState.WAITING else None)
    checkpoint = RecoveryCheckpoint(execution.execution_id,
        None if state is RecoveryCheckpointState.ADMITTED else "run", state, 0,
        execution.created_at, execution.updated_at, pending_tools=pending)
    checked = []
    mutations = []

    class Port:
        tenant_id = "default"

        async def load_execution(self, execution_id: str, *, tenant_id: str) -> ExecutionRecord:
            return execution

        async def load_recovery_checkpoint(self, execution_id: str, *, tenant_id: str) -> RecoveryCheckpoint:
            return checkpoint

        def validate_binding(self, value: ExecutionRecord) -> None:
            assert value is execution

        async def validate_recovery_inputs(self, value: ExecutionRecord) -> None:
            checked.append(value.tool_response_ref)
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)

        async def _reconcile_tool_effects(self, *args: object, **kwargs: object) -> tuple[tuple[object, ...], int]:
            return (), 0

        async def _pending_cancel_operations(self, *args: object, **kwargs: object) -> tuple[object, ...]:
            return ()

        async def _reconcile_session_recovery(self, *args: object, **kwargs: object) -> bool:
            return True

        async def _recovery_idempotency(self, value: ExecutionRecord) -> object:
            return object()

        def _reset_local_producer(self, execution_id: str) -> None:
            pass

        async def _unexpected_mutation(self, *args: object, **kwargs: object) -> None:
            mutations.append("resume")
            raise AssertionError("unreadable fixture must not advance recovery")

        _commit_start_recovery_checkpoint = _unexpected_mutation
        _commit_recovery_resume = _unexpected_mutation
        claim_deferred_resume = _unexpected_mutation

        async def _complete_recovered_cancel(self, *args: object, **kwargs: object) -> ExecutionRecord:
            mutations.append("cancel")
            return replace(execution, status=ExecutionStatus.CANCELLED)

        async def _finish_checkpoint(self, value: RecoveryCheckpoint) -> None:
            mutations.append("terminal-cleanup")

        async def load_external_call(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                status=ExternalCallStatus.PENDING if path == "pending_deferred" else ExternalCallStatus.SUPPLIED,
                result_payload=StoredPayload.inline_text("answer"), resolution_kind="succeeded", resolution_metadata={},
            )

        async def read_deferred_payload(self, payload: StoredPayload) -> object:
            return payload.decode()

        async def load_interrupted_messages(self, agent_run_id: str) -> tuple[object, ...]:
            return ()

    coordinator = _RecoveryCoordinator(Port(), None)
    if path == "terminal_cleanup":
        assert await coordinator.reconcile_checkpoint(checkpoint) is False
        assert checked == [] and mutations == ["terminal-cleanup"]
    elif path == "cancel_recovery":
        result, launched = await coordinator.recover_execution(execution.execution_id,
            tenant_id="default", expected_revision=execution.revision)
        assert result.status is ExecutionStatus.CANCELLED and launched is False
        assert checked == [] and mutations == ["cancel"]
    elif path == "pending_deferred":
        assert await coordinator.reconcile_waiting_deferred(checkpoint, execution) is None
        assert checked == [] and mutations == []
    else:
        with pytest.raises(AIError) as unavailable:
            if path == "explicit_recovery":
                await coordinator.recover_execution(execution.execution_id,
                    tenant_id="default", expected_revision=execution.revision)
            elif path == "ready_deferred":
                await coordinator.reconcile_waiting_deferred(checkpoint, execution)
            else:
                await coordinator.reconcile_checkpoint(checkpoint)
        assert unavailable.value.code is ErrorCode.ASSET_VERSION_NOT_FOUND
        assert checked == [execution.tool_response_ref] and mutations == []


@pytest.mark.asyncio
async def test_fixture_disappearing_after_admission_preserves_execution_for_repaired_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, first, _second = await _fixtures()
    storage = RuntimeStorage.filesystem(tmp_path)
    unavailable = False
    read_versions = assets.read_versions

    async def read_fixture(refs: Sequence[AssetVersionRef]) -> tuple[bytes, ...]:
        if unavailable:
            raise AIError(ErrorCode.ASSET_VERSION_NOT_FOUND)
        return await read_versions(refs)

    monkeypatch.setattr(assets, "read_versions", read_fixture)
    async with Runtime.open("fixture-repair", models=RuntimeUsageModels(), storage=storage,
                            tool_responses=ToolResponseFixture(first, assets)) as runtime:
        reserve = storage.execution.executions.reserve_start

        async def remove_after_admission(value: ExecutionStartReservation) -> object:
            nonlocal unavailable
            admitted = await reserve(value)
            unavailable = True
            return admitted

        with monkeypatch.context() as patch:
            patch.setattr(storage.execution.executions, "reserve_start", remove_after_admission)
            execution = await runtime.agents.get().start("repair the same accepted execution")
        with pytest.raises(AIError) as missing:
            await execution.wait(timeout_seconds=10)
        assert missing.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
        saved = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert saved.status is ExecutionStatus.STARTED
        assert saved.tool_response_ref == first and saved.result is None
        unavailable = False
        recovered = await execution.recover(idempotency_key="repair-fixture")
        assert recovered.execution_id == execution.execution_id
        assert (await recovered.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        terminal = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert terminal.tool_response_ref == first
    await assets.close()
