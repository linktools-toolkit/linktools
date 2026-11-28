#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
from linktools.ai.task.engine import TaskEngine
"""Conversation guarantees are fixed at admission and preserved through recovery."""
from ._runtime_test_helpers import configure_execution
from ._runtime_test_helpers import model_features

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
    decode_envelope,
    encode_domain,
    encode_envelope,
)
from linktools.ai.persistence.contracts import ExecutionRecord
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from ._runtime_test_helpers import RuntimeUsageModels
from .test_runtime_binding_contract import _execution


@pytest.mark.asyncio
async def test_conversation_retention_does_not_register_a_capability() -> None:
    group = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})
    capture = await group.resolve()
    assert not capture.contributions
    assert capture.defaults["execution"] == {"retain_history": True}


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
def test_execution_wire_uses_declared_conversation_default(persisted: bool) -> None:
    record = _execution()
    assert record.conversation is True
    payload = (_encode_persisted_domain if persisted else encode_domain)(record)
    payload["fields"].pop("conversation")
    restored = (
        _decode_enveloped_domain(
            encode_envelope({"type": "execution_record", "payload": payload}), ExecutionRecord,
        ) if persisted else decode_domain(payload, ExecutionRecord)
    )
    assert restored == record


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
    group: RuntimeFeatures[None] = RuntimeFeatures("application", defaults={"execution": {"retain_history": conversation}})
    group.agent("default", model="default", allow_capabilities=())
    policies: list[AbstractCapability[AgentContext[None]]] = []
    if restart:
        policies.append(native_checkpoint())
    if trace:
        policies.append(TraceCapability())
    if combined:
        group.capability(CombinedCapability(policies), id="runtime-persistence", scope="runtime")
    else:
        for capability in policies:
            group.capability(capability, scope="runtime")
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open('conversation-feature', storage=storage) as persistence_159, Runtime.open(persistence=persistence_159, features=model_features(RuntimeUsageModels(), group)) as runtime:
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
    group: RuntimeFeatures[None] = RuntimeFeatures("application", defaults={"execution": {"retain_history": True}})
    storage = PersistenceStorage.in_memory()
    async with Persistence.open('conversation-feature', storage=storage) as persistence_180, Runtime.open(persistence=persistence_180, features=model_features(RuntimeUsageModels(), group)) as runtime, TaskEngine.open(persistence=persistence_180) as engine:
        graph = await engine.with_definitions(task).start(
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
            configure_execution(group, retain_history=True)
        storage = PersistenceStorage.sqlite(database)
        async with Persistence.open('conversation-feature', storage=storage) as persistence_207, Runtime.open(persistence=persistence_207, features=model_features(RuntimeUsageModels(), group)) as runtime:
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
    from linktools.ai.execution import ExecutionRequest

    group: RuntimeFeatures[None] = RuntimeFeatures("application", defaults={"execution": {"auto_recover": True}})
    group.capability(native_checkpoint(), scope="runtime")
    configure_execution(group, retain_history=True)
    storage_path = tmp_path / "runtime"
    storage = PersistenceStorage.from_root(storage_path, exclusive_writer=True)

    async def interrupted_launch(
        request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
    ) -> None:
        del request, execution, kwargs

    async with Persistence.open('conversation-feature', storage=storage) as persistence_247, Runtime.open(persistence=persistence_247, features=model_features(RuntimeUsageModels(), group)) as runtime:
        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            execution = await runtime.agents.get().start("answer", idempotency_key="answer")
        repository = persistence_247.execution.executions
        before = await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert before is not None and before.status is ExecutionStatus.STARTED
        assert before.restart_checkpoints and before.conversation
        written = await repository.state_store.read(
            lambda transaction: transaction.get_record(repository.execution_record_key(execution.execution_id))
        )
        assert written is not None
        fields = decode_envelope(written.data).value["payload"]["fields"]
        assert "conversation" not in fields and "runtime_policy" in fields
        expanded, = await repository._execution_content.expand_policies((written,))
        assert decode_envelope(expanded.data).value["payload"]["fields"]["conversation"] is True
        assert await repository._execution_content.decode(written) == before

    reopened = PersistenceStorage.from_root(storage_path, exclusive_writer=True)
    async with Persistence.open('conversation-feature', storage=reopened) as persistence_260, Runtime.open(persistence=persistence_260, features=model_features(RuntimeUsageModels(), group)) as runtime:
        recovered = await runtime.executions.get(execution.execution_id)
        assert (await recovered.wait(timeout_seconds=10)).result.output == {"text": "done"}
        after = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert after is not None and after.status is ExecutionStatus.SUCCEEDED
        assert after.restart_checkpoints and after.conversation
