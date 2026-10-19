#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conversation guarantees are fixed at admission and preserved through recovery."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import TraceCapability, CheckpointCapability, ConversationCapability
from linktools.ai.runtime import Runtime, RuntimeStorage
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


def _conversation_feature(combined: bool) -> AbstractCapability[AgentContext[None]]:
    capability = ConversationCapability()
    assert isinstance(capability, AbstractCapability)
    return CombinedCapability([capability]) if combined else capability


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", (False, True), ids=("native", "combined"))
async def test_conversation_uses_native_runtime_registration(combined: bool) -> None:
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.capability(_conversation_feature(combined), id="conversation-feature" if combined else None, scope="runtime")
    capture = await group.capture()
    assert len(capture.contributions) == 1
    contribution = capture.contributions[0]
    assert contribution.scope == "runtime"
    captured: list[AbstractCapability[object]] = []
    contribution.value.apply(captured.append)
    assert any(isinstance(value, ConversationCapability) for value in captured)


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", (False, True), ids=("native", "combined"))
async def test_conversation_rejects_agent_scope(combined: bool) -> None:
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.capability(_conversation_feature(combined), id="conversation-feature" if combined else None, scope="agent")
    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            "conversation-feature", models=RuntimeUsageModels(),
            storage=RuntimeStorage.in_memory(), features=group,
        ):
            raise AssertionError("Agent-scoped Conversation was accepted")
    assert invalid.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_conversation_rejects_duplicate_native_id_across_contributions() -> None:
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    for policy_id in ("first-policy", "second-policy"):
        group.capability(
            CombinedCapability([ConversationCapability()]),
            id=policy_id, scope="runtime",
        )
    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            "conversation-feature", models=RuntimeUsageModels(),
            storage=RuntimeStorage.in_memory(), features=group,
        ):
            raise AssertionError("Duplicate Conversation capabilities were accepted")
    assert invalid.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
@pytest.mark.parametrize("conversation", (False, True))
def test_conversation_wire_roundtrip_is_explicit(persisted: bool, conversation: bool) -> None:
    record = replace(_execution(), conversation=conversation)
    payload = (_encode_persisted_domain if persisted else encode_domain)(record)
    assert payload["fields"]["conversation"] is conversation
    restored = (
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        ) if persisted else decode_domain(payload, ExecutionRecord)
    )
    assert restored == record


@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
def test_execution_wire_requires_conversation_guarantee(persisted: bool) -> None:
    record = _execution()
    assert record.conversation is True
    payload = (_encode_persisted_domain if persisted else encode_domain)(record)
    payload["fields"].pop("conversation")
    with pytest.raises(AIError) as missing:
        if persisted:
            _decode_enveloped_domain(
                encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
            )
        else:
            decode_domain(payload, ExecutionRecord)
    assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert missing.value.safe_details["missing_fields"] == ["conversation"]


@pytest.mark.parametrize("invalid", (1, "true", None))
def test_conversation_rejects_malformed_wire_and_constructor_values(invalid: object) -> None:
    record = _execution()
    with pytest.raises(TypeError, match="persistence guarantees"):
        replace(record, conversation=invalid)
    payload = _encode_persisted_domain(record)
    payload["fields"]["conversation"] = invalid
    with pytest.raises(AIError) as error:
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation", "restart", "trace", "combined"),
    (
        (False, False, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (True, True, False, False),
        (True, False, True, True),
        (True, True, True, True),
    ),
    ids=("default", "conversation", "combined", "restart", "trace", "restart-and-trace"),
)
async def test_admitted_policy_preserves_stronger_durability(
    tmp_path: Path, conversation: bool, restart: bool, trace: bool, combined: bool,
) -> None:
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.agent("default", model="default", allow_capabilities=())
    policies: list[AbstractCapability[AgentContext[None]]] = []
    if conversation:
        policies.append(ConversationCapability())
    if restart:
        policies.append(CheckpointCapability())
    if trace:
        policies.append(TraceCapability())
    if combined:
        group.capability(CombinedCapability(policies), id="runtime-persistence", scope="runtime")
    else:
        for capability in policies:
            group.capability(capability, scope="runtime")
    storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    async with Runtime.open(
        "conversation-feature", models=RuntimeUsageModels(), storage=storage, features=group,
    ) as runtime:
        execution = await runtime.agents.get().start("answer", idempotency_key="answer")
        assert (await execution.wait(timeout_seconds=10)).result.output == {"text": "done"}
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None
        assert record.conversation is conversation
        assert record.restart_checkpoints is restart
        assert record.durable_trace is trace


@pytest.mark.asyncio
async def test_task_admission_keeps_conversation_disabled() -> None:
    async def answer(_context: TaskNodeContext[None]) -> str:
        return "done"

    task = Task("answer", answer, effect_policy="none")
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.capability(ConversationCapability(), scope="runtime")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(
        "conversation-feature", models=RuntimeUsageModels(), storage=storage, features=group,
    ) as runtime:
        graph = await runtime.tasks.bind(task).start(
            TaskGraph("task-policy", (TaskNode("answer", task=task),)), idempotency_key="task-policy",
        )
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        execution = await graph.execution("answer")
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.binding_kind == "task"
        assert record.conversation is False
        assert record.restart_checkpoints and record.durable_trace


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_conversation", (False, True))
async def test_reopening_cannot_reinterpret_admitted_conversation(
    tmp_path: Path, initial_conversation: bool,
) -> None:
    database = tmp_path / "runtime.db"
    execution_id: str | None = None
    for enabled in (initial_conversation, not initial_conversation):
        group: RuntimeFeatures[None] = RuntimeFeatures("application")
        group.agent("default", model="default", allow_capabilities=())
        if enabled:
            group.capability(ConversationCapability(), scope="runtime")
        storage = RuntimeStorage.sqlite(database)
        async with Runtime.open(
            "conversation-feature", models=RuntimeUsageModels(), storage=storage, features=group,
        ) as runtime:
            execution = await runtime.agents.get().start("answer", idempotency_key="answer")
            assert (await execution.wait(timeout_seconds=10)).result.output == {"text": "done"}
            if execution_id is None:
                execution_id = execution.execution_id
            assert execution.execution_id == execution_id
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.conversation is initial_conversation


@pytest.mark.asyncio
async def test_current_execution_record_retains_conversation_through_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime.service_api import ExecutionRequest
    from linktools.ai.runtime.state import _repository_common

    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.capability(CheckpointCapability(), scope="runtime")
    group.capability(ConversationCapability(), scope="runtime")
    storage_path = tmp_path / "runtime"
    storage = RuntimeStorage.filesystem(storage_path)
    encode = _repository_common._encode_persisted_domain
    written: list[JsonValue] = []

    def capture_current_record(value: object) -> JsonValue:
        payload = encode(value)
        if isinstance(value, ExecutionRecord):
            assert payload["fields"]["conversation"] is True
            written.append(payload)
        return payload

    async def interrupted_launch(
        request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
    ) -> None:
        del request, execution, kwargs

    async with Runtime.open(
        "conversation-feature", models=RuntimeUsageModels(), storage=storage, features=group,
    ) as runtime:
        with monkeypatch.context() as fault:
            fault.setattr(_repository_common, "_encode_persisted_domain", capture_current_record)
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            execution = await runtime.agents.get().start("answer", idempotency_key="answer")
        assert written
        before = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert before is not None and before.status is ExecutionStatus.STARTED
        assert before.restart_checkpoints and before.conversation

    reopened = RuntimeStorage.filesystem(storage_path)
    async with Runtime.open(
        "conversation-feature", models=RuntimeUsageModels(), storage=reopened, features=group,
    ) as runtime:
        recovered = await runtime.executions.get(execution.execution_id)
        assert (await recovered.wait(timeout_seconds=10)).result.output == {"text": "done"}
        after = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert after is not None and after.status is ExecutionStatus.SUCCEEDED
        assert after.restart_checkpoints and after.conversation
