#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal commit is a captured native Runtime policy with explicit wire meaning."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    DurableTraceCapability,
    RestartCheckpointCapability,
    Runtime,
    RuntimeStorage,
    TerminalCommitCapability,
)
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from ._runtime_test_helpers import RuntimeUsageModels
from .test_runtime_binding_contract import _execution


def _terminal_policy(combined: bool) -> AbstractCapability[AgentContext[None]]:
    capability = TerminalCommitCapability()
    assert isinstance(capability, AbstractCapability)
    return CombinedCapability([capability]) if combined else capability


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", (False, True), ids=("native", "combined"))
async def test_terminal_commit_uses_native_runtime_registration(combined: bool) -> None:
    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.capability(_terminal_policy(combined), id="terminal-policy" if combined else None, scope="runtime")
    capture = await group.capture()
    assert len(capture.contributions) == 1
    contribution = capture.contributions[0]
    assert contribution.scope == "runtime"
    captured: list[AbstractCapability[object]] = []
    contribution.value.apply(captured.append)
    assert any(isinstance(value, TerminalCommitCapability) for value in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", (False, True), ids=("native", "combined"))
async def test_terminal_commit_rejects_agent_scope(combined: bool) -> None:
    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.capability(_terminal_policy(combined), id="terminal-policy" if combined else None)
    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            "terminal-policy", models=RuntimeUsageModels(),
            storage=RuntimeStorage.in_memory(), capabilities=(group,),
        ):
            raise AssertionError("Agent-scoped terminal commit policy was accepted")
    assert invalid.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_terminal_commit_rejects_duplicate_native_id_across_groups() -> None:
    first: CapabilityGroup[None] = CapabilityGroup("first")
    second: CapabilityGroup[None] = CapabilityGroup("second")
    for group in (first, second):
        group.capability(
            CombinedCapability([TerminalCommitCapability()]),
            id=f"{group.id}-policy", scope="runtime",
        )
    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            "terminal-policy", models=RuntimeUsageModels(),
            storage=RuntimeStorage.in_memory(), capabilities=(first, second),
        ):
            raise AssertionError("Duplicate terminal commit policies were accepted")
    assert invalid.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
@pytest.mark.parametrize("terminal_commit", (False, True))
def test_terminal_commit_wire_roundtrip_is_explicit(persisted: bool, terminal_commit: bool) -> None:
    record = replace(_execution(), terminal_commit=terminal_commit)
    payload = (_encode_persisted_domain if persisted else encode_domain)(record)
    assert payload["fields"]["terminal_commit"] is terminal_commit
    restored = (
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        ) if persisted else decode_domain(payload, ExecutionRecord)
    )
    assert restored == record


@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
def test_earlier_execution_record_keeps_durable_path(persisted: bool) -> None:
    record = _execution()
    assert record.terminal_commit is False
    payload = (_encode_persisted_domain if persisted else encode_domain)(record)
    payload["fields"].pop("terminal_commit")
    restored = (
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        ) if persisted else decode_domain(payload, ExecutionRecord)
    )
    assert restored == record
    assert restored.terminal_commit is False


@pytest.mark.parametrize("invalid", (1, "true", None))
def test_terminal_commit_rejects_malformed_wire_and_constructor_values(invalid: object) -> None:
    record = _execution()
    with pytest.raises(TypeError, match="persistence guarantees"):
        replace(record, terminal_commit=invalid)
    payload = _encode_persisted_domain(record)
    payload["fields"]["terminal_commit"] = invalid
    with pytest.raises(AIError) as error:
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_commit", "restart", "trace", "combined"),
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, True, True, True),
    ),
    ids=("default", "terminal", "combined", "restart", "trace", "restart-and-trace"),
)
async def test_admitted_policy_preserves_stronger_durability(
    tmp_path: Path, terminal_commit: bool, restart: bool, trace: bool, combined: bool,
) -> None:
    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_capabilities=())
    policies: list[AbstractCapability[AgentContext[None]]] = []
    if terminal_commit:
        policies.append(TerminalCommitCapability())
    if restart:
        policies.append(RestartCheckpointCapability())
    if trace:
        policies.append(DurableTraceCapability())
    if combined:
        group.capability(CombinedCapability(policies), id="runtime-persistence", scope="runtime")
    else:
        for capability in policies:
            group.capability(capability, scope="runtime")
    storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    async with Runtime.open(
        "terminal-policy", models=RuntimeUsageModels(), storage=storage, capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get().start("answer", idempotency_key="answer")
        assert (await execution.wait(timeout_seconds=10)).result.output == {"text": "done"}
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None
        assert record.terminal_commit is (terminal_commit and not (restart or trace))
        assert record.restart_checkpoints is restart
        assert record.durable_trace is trace


@pytest.mark.asyncio
async def test_task_admission_keeps_terminal_commit_disabled() -> None:
    async def answer(_context: TaskNodeContext[None]) -> str:
        return "done"

    task = Task("answer", answer, effect_policy="none")
    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.capability(TerminalCommitCapability(), scope="runtime")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "terminal-policy", models=RuntimeUsageModels(), storage=storage, capabilities=(group,),
    ) as runtime:
        graph = await runtime.tasks.bind(task).start(
            TaskGraph("task-policy", (TaskNode("answer", task=task),)), idempotency_key="task-policy",
        )
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        execution = await graph.execution("answer")
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.binding_kind == "task"
        assert record.terminal_commit is False
        assert record.restart_checkpoints and record.durable_trace


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_terminal_commit", (False, True))
async def test_reopening_cannot_reinterpret_admitted_terminal_commit(
    tmp_path: Path, initial_terminal_commit: bool,
) -> None:
    database = tmp_path / "runtime.db"
    execution_id: str | None = None
    for enabled in (initial_terminal_commit, not initial_terminal_commit):
        group: CapabilityGroup[None] = CapabilityGroup("application")
        group.agent("default", model="default", allow_capabilities=())
        if enabled:
            group.capability(TerminalCommitCapability(), scope="runtime")
        storage = RuntimeStorage.sqlite(database)
        async with Runtime.open(
            "terminal-policy", models=RuntimeUsageModels(), storage=storage, capabilities=(group,),
        ) as runtime:
            execution = await runtime.agents.get().start("answer", idempotency_key="answer")
            assert (await execution.wait(timeout_seconds=10)).result.output == {"text": "done"}
            if execution_id is None:
                execution_id = execution.execution_id
            assert execution.execution_id == execution_id
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.terminal_commit is initial_terminal_commit


@pytest.mark.asyncio
async def test_earlier_execution_record_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.service_api import ExecutionRequest
    from linktools.ai.runtime.state import _repository_common

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.capability(RestartCheckpointCapability(), scope="runtime")
    storage_path = tmp_path / "runtime"
    storage = RuntimeStorage.filesystem(storage_path)
    encode = _repository_common._encode_persisted_domain
    written: list[JsonValue] = []

    def encode_earlier_record(value: object) -> JsonValue:
        payload = encode(value)
        if isinstance(value, ExecutionRecord):
            payload["fields"].pop("terminal_commit")
            written.append(payload)
        return payload

    async def interrupted_launch(
        request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
    ) -> None:
        del request, execution, kwargs

    async with Runtime.open(
        "terminal-policy", models=RuntimeUsageModels(), storage=storage, capabilities=(group,),
    ) as runtime:
        with monkeypatch.context() as fault:
            fault.setattr(_repository_common, "_encode_persisted_domain", encode_earlier_record)
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            execution = await runtime.agents.get().start("answer", idempotency_key="answer")
        assert written
        before = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert before is not None and before.status is ExecutionStatus.STARTED
        assert before.restart_checkpoints and not before.terminal_commit

    reopened = RuntimeStorage.filesystem(storage_path)
    async with Runtime.open(
        "terminal-policy", models=RuntimeUsageModels(), storage=reopened, capabilities=(group,),
    ) as runtime:
        recovered = await runtime.executions.get(execution.execution_id)
        assert (await recovered.wait(timeout_seconds=10)).result.output == {"text": "done"}
        after = await reopened.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert after is not None and after.status is ExecutionStatus.SUCCEEDED
        assert after.restart_checkpoints and not after.terminal_commit
