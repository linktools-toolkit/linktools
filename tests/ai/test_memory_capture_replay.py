#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Imported memory preserves captured versions and replays without rollback."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.memory import MemoryOperation, MemoryOperationConflictError

from linktools.ai.capability import AgentContext, ExecutionCapability, MemoryCapability, StepPersistenceCapability
from linktools.ai.core import JsonValue, OperationKind, OperationLedgerInput, OperationLedgerRecord, OperationStatus, Principal, ResourceKind, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext, ExternalCallSucceeded, ExternalSupplyRequest
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.contracts import MemoryRecord
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._memory import RuntimeMemoryStore, _memory_id
from linktools.ai.core import ExecutionStatus

from ._runtime_test_helpers import _Models, _UsageFunctionModel


_PATH = "memory/MEMORY.md"


def _store(state: Persistence, scope: str) -> RuntimeMemoryStore:
    return RuntimeMemoryStore(
        state.memory, object_store=state.object_store(PersistenceDomain.MEMORY),
        namespace=state.namespace, tenant_id=state.tenant_id,
        execution_id="execution", memory_scope=scope,
    )


def _storage(backend: str, root: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(root / "state.db")


async def _capture(state: Persistence) -> Mapping[str, JsonValue]:
    source = _store(state, "source")
    await source.write(_PATH, "source note", expected_version=None)
    return await source.capture()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_memory_capture_replay_preserves_source_and_later_mutations(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("memory-capture", storage=_storage(backend, tmp_path)) as state:
        snapshot = await _capture(state)
        target = _store(state, "target")
        await target.seed_capture(snapshot)
        initial = await target.read(_PATH, max_chars=100)
        source = await _store(state, "source").read(_PATH, max_chars=100)
        assert initial is not None and source is not None
        assert (initial.content, initial.version) == (source.content, source.version)
        await target.seed_capture(snapshot)
        assert await target.read(_PATH, max_chars=100) == initial
        updated = await target.write(_PATH, "target update", expected_version=initial.version)
        current = await target.read(_PATH, max_chars=100)
        await target.seed_capture(snapshot)
        assert await target.read(_PATH, max_chars=100) == current
        await target.delete(_PATH, expected_version=updated.version)
        await target.seed_capture(snapshot)
        assert await target.read(_PATH, max_chars=100) is None
        for changed in (
            {_PATH: {"content": "different", "version": initial.version}},
            {_PATH: {"content": initial.content, "version": updated.version}},
        ):
            with pytest.raises(MemoryOperationConflictError):
                await target.seed_capture(changed)
        assert await target.read(_PATH, max_chars=100) is None


@pytest.mark.asyncio
async def test_memory_capture_receipt_replays_after_sqlite_reopen(tmp_path: Path) -> None:
    storage = _storage("sqlite", tmp_path)
    async with Persistence.open("memory-capture", storage=storage) as state:
        snapshot = await _capture(state)
        target = _store(state, "target")
        await target.seed_capture(snapshot)
        before = await target.read(_PATH, max_chars=100)
    async with Persistence.open("memory-capture", storage=storage) as state:
        target = _store(state, "target")
        await target.seed_capture(snapshot)
        assert await target.read(_PATH, max_chars=100) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_memory_capture_retry_resolves_committed_partial_seed(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    async with Persistence.open("memory-capture", storage=_storage(backend, tmp_path)) as state:
        source = _store(state, "source")
        await source.write(_PATH, "source note", expected_version=None)
        await source.write("memory/other.md", "other note", expected_version=None)
        snapshot = await source.capture()
        target = _store(state, "target")
        apply_write = state.memory.records.apply_write

        async def lose_acknowledgment(
            record: MemoryRecord, *, expected_revision: int | None, operation: OperationLedgerInput,
        ) -> tuple[MemoryRecord | None, bool]:
            await apply_write(record, expected_revision=expected_revision, operation=operation)
            if cancelled:
                raise asyncio.CancelledError()
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(state.memory.records, "apply_write", lose_acknowledgment)
            with pytest.raises(asyncio.CancelledError if cancelled else AIError) as raised:
                await target.seed_capture(snapshot)
            if not cancelled:
                assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        initial = await target.read(_PATH, max_chars=100)
        assert initial is not None
        await target.write(_PATH, "new target note", expected_version=initial.version)
        await target.seed_capture(snapshot)
        current = await target.read(_PATH, max_chars=100)
        other = await target.read("memory/other.md", max_chars=100)
        assert current is not None and current.content == "new target note"
        assert other is not None and other.content == "other note"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["version", "path", "kind", "fingerprint", "owner", "resource", "status", "sequence", "format"])
async def test_memory_capture_receipt_rejects_inconsistent_replay(
    corruption: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("memory-capture", storage=PersistenceStorage.in_memory()) as state:
        snapshot = await _capture(state)
        target = _store(state, "target")
        await target.seed_capture(snapshot)
        initial = await target.read(_PATH, max_chars=100)
        assert initial is not None and initial.operation_id is not None
        record = await state.memory.operations.get(initial.operation_id, tenant_id=state.tenant_id)
        assert record is not None and record.result_ref is not None
        payload = json.loads(record.result_ref)
        expected_error = AIError
        if corruption == "version":
            payload["result"]["version"] = "m1:" + "0" * 64
        elif corruption == "path":
            payload["result"]["file"] = "memory/other.md"
            record = replace(record, resource_id=_memory_id(canonical_sha256("target"), "memory/other.md"))
            expected_error = MemoryOperationConflictError
        elif corruption == "kind":
            payload["result"].update(version=None, status="deleted")
            record = replace(record, operation_kind=OperationKind.MEMORY_DELETE)
            expected_error = MemoryOperationConflictError
        elif corruption == "fingerprint":
            record = replace(record, request_digest="0" * 64)
            expected_error = MemoryOperationConflictError
        elif corruption == "owner":
            record = replace(record, operation_id="wrong-owner")
        elif corruption == "resource":
            record = replace(record, resource_kind=ResourceKind.ARTIFACT)
        elif corruption == "status":
            record = replace(record, status=OperationStatus.RUNNING)
        elif corruption == "sequence":
            record = replace(record, sequence=0)
        else:
            payload["version"] = 2
        corrupt = replace(record, result_ref=json.dumps(payload))

        async def read_corrupt(operation_id: str, *, tenant_id: str) -> OperationLedgerRecord:
            return corrupt

        with monkeypatch.context() as patch:
            patch.setattr(state.memory.operations, "get", read_corrupt)
            with pytest.raises(expected_error) as raised:
                await target.seed_capture(snapshot)
            if isinstance(raised.value, AIError):
                assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await target.read(_PATH, max_chars=100) == initial


@pytest.mark.asyncio
async def test_memory_ordinary_receipt_still_requires_target_version(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("memory-capture", storage=PersistenceStorage.in_memory()) as state:
        snapshot = await _capture(state)
        target = _store(state, "target")
        operation = MemoryOperation("capture:ordinary", "a" * 64)
        await target.write(_PATH, "target note", expected_version=None, operation=operation)
        initial = await target.read(_PATH, max_chars=100)
        assert initial is not None and initial.operation_id is not None
        record = await state.memory.operations.get(initial.operation_id, tenant_id=state.tenant_id)
        assert record is not None and record.result_ref is not None
        payload = json.loads(record.result_ref)
        payload["result"]["version"] = snapshot[_PATH]["version"]
        corrupt = replace(record, result_ref=json.dumps(payload))

        async def read_corrupt(operation_id: str, *, tenant_id: str) -> OperationLedgerRecord:
            return corrupt

        with monkeypatch.context() as patch:
            patch.setattr(state.memory.operations, "get", read_corrupt)
            with pytest.raises(AIError) as raised:
                await target.get_operation(operation)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            with pytest.raises(AIError) as raised:
                await target.write(_PATH, "target note", expected_version=None, operation=operation)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await target.read(_PATH, max_chars=100) == initial


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_mode", ["empty", "captured", "updated"])
async def test_runtime_imported_memory_resumes_deferred_without_rollback(tmp_path: Path, memory_mode: str) -> None:
    async with Persistence.open("memory-source", storage=PersistenceStorage.in_memory()) as state:
        snapshot = {} if memory_mode == "empty" else await _capture(state)
    observed: list[str] = []

    async def deferred(context: AgentContext[None]) -> str:
        raise CallDeferred()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = {part.tool_name: part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)}
        if "deferred" not in returns:
            if memory_mode == "updated" and "write_memory" not in returns:
                return ModelResponse(parts=[ToolCallPart("write_memory", {"file": "MEMORY.md", "content": "target update", "old_text": "source note"}, tool_call_id="write-one")])
            if "write_memory" in returns:
                assert returns["write_memory"].outcome != "retried"
            return ModelResponse(parts=[ToolCallPart("deferred", {}, tool_call_id="defer-one")])
        if memory_mode != "empty":
            if "read_memory" not in returns:
                return ModelResponse(parts=[ToolCallPart("read_memory", {"file": "MEMORY.md"}, tool_call_id="read-one")])
            assert returns["read_memory"].outcome != "retried"
            observed.append(str(returns["read_memory"].content))
        return ModelResponse(parts=[TextPart("finished")])

    features = RuntimeFeatures(workspace_factory=None)
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(MemoryCapability())
    features.tool(deferred)
    async with Runtime.open("memory-resume", storage=_storage("sqlite", tmp_path), features=features, auto_recover=False) as runtime:
        execution = await runtime.agents.get().start("begin", input_context=ExecutionInputContext.from_messages((), memory=snapshot))
        principal = runtime.default_principal
        async with asyncio.timeout(5):
            while True:
                current = await runtime.executions.inspect(execution.execution_id, principal=principal)
                if current.status in (ExecutionStatus.WAITING_DEFERRED, ExecutionStatus.FAILED, ExecutionStatus.SUCCEEDED):
                    break
                await asyncio.sleep(0.01)
        assert current.status is ExecutionStatus.WAITING_DEFERRED
        with pytest.raises(AIError) as raised:
            await runtime.external_calls.list(execution.execution_id, principal=principal)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        resolver = Principal("resolver", runtime.tenant_id, "service")
        pending = await runtime.external_calls.list(execution.execution_id, principal=resolver)
        assert len(pending) == 1
        await runtime.external_calls.supply(execution.execution_id, ExternalSupplyRequest(
            resolver, pending[0].call_id, "supply-one", ExternalCallSucceeded("external result"),
        ))
        outcome = await runtime.executions.wait(execution.execution_id, principal=principal, timeout_seconds=5)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED
    if memory_mode != "empty":
        assert len(observed) == 1
        assert ("target update" if memory_mode == "updated" else "source note") in observed[0]
        if memory_mode == "updated":
            assert "source note" not in observed[0]
