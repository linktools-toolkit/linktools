#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restart checkpoints and durable trace are independent Runtime opt-ins."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    DurableTraceCapability,
    RestartCheckpointCapability,
    Runtime,
    RuntimeStorage,
)
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import (
    ExecutionHistoryState,
    ExecutionRecord,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from ._session_tool_test_helpers import _split_sqlite_storage
from .test_runtime_binding_contract import _execution


_NAMESPACE = "optional-persistence"


class _Answer(BaseModel):
    answer: str
    evidence: dict[str, int]


async def _start_delegated_execution(
    runtime: Runtime, storage: RuntimeStorage, parent_id: str, prompt: str,
) -> str:
    from linktools.ai.runtime import ExecutionRequest

    parent = await storage.execution.executions.get(parent_id, tenant_id=runtime.tenant_id)
    assert parent is not None
    handle = await runtime._execution_service.start_subagent(
        parent.binding_digest,
        ExecutionRequest(prompt, runtime.default_principal, prompt, None, "run", False, False),
        parent_execution_id=parent.execution_id, root_execution_id=parent.root_execution_id,
        parent_invocation_id=f"delegate-{prompt}", binding_contract=parent.binding,
    )
    return handle.execution_id


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True), ids=("no-restart", "restart"))
@pytest.mark.parametrize("trace", (False, True), ids=("no-trace", "trace"))
@pytest.mark.parametrize("split_storage", (False, True), ids=("combined-storage", "split-storage"))
async def test_persistence_capabilities_preserve_sessions_and_results_independently(
    tmp_path: Path, restart: bool, trace: bool, split_storage: bool,
) -> None:
    tool_calls: list[str] = []
    model_calls: list[str] = []
    ready = asyncio.Event()
    release = asyncio.Event()
    first_output = {"answer": "first answer", "evidence": {"value": 42}}
    second_output = {"answer": "context restored", "evidence": {"value": 42}}

    async def lookup(_context: AgentContext[None]) -> str:
        tool_calls.append("lookup")
        return "retained tool result"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [
            part.content for message in messages for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        returned = [
            part for message in messages for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "lookup"
        ]
        prompt = prompts[-1]
        assert isinstance(prompt, str)
        model_calls.append(prompt)
        if prompt == "second question":
            assert "first question" in prompts
            assert [part.content for part in returned] == ["retained tool result"]
            output = second_output
        elif not returned:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="lookup-call")])
        else:
            ready.set()
            await release.wait()
            output = first_output
        assert info.output_tools
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, output)])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.tool(lookup, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("lookup",), allow_capabilities=())
    if restart:
        group.capability(RestartCheckpointCapability(), scope="runtime")
    if trace:
        group.capability(DurableTraceCapability(), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "runtime.db"
    storage_factory = _split_sqlite_storage if split_storage else RuntimeStorage.sqlite
    storage = storage_factory(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=storage, capabilities=(group,),
    ) as runtime:
        await runtime.agents.get().create_session("session")
        session = runtime.agents.get().session("session")
        execution = await session.start("first question", output=_Answer, idempotency_key="first")
        try:
            await asyncio.wait_for(ready.wait(), 10)
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.restart_checkpoints is restart
            assert record.durable_trace is trace
            run_id = agent_run_id(
                namespace=_NAMESPACE, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=record.agent_run_seq,
            )
            checkpoint = await storage.run_store.read_store(RuntimeDomain.RECOVERY).latest_checkpoint(
                agent_run_id=run_id, include_interrupted=True,
            )
            assert (checkpoint is not None) is restart
            active_head = await storage.execution.executions.get_history_head(
                execution.execution_id, tenant_id=runtime.tenant_id,
            )
            assert active_head is not None
            assert active_head.state is ExecutionHistoryState.OPEN
            assert active_head.producer_generation > 0
            assert active_head.producer_claim_id is not None

            async def observe_progress() -> None:
                seen: set[ExecutionEventType] = set()
                stream = execution.watch()
                try:
                    async for item in stream:
                        seen.add(item.event.event_type)
                        if {ExecutionEventType.MODEL_REQUEST_STARTED, ExecutionEventType.TOOL_CALL_FINISHED} <= seen:
                            return
                finally:
                    await stream.aclose()

            await asyncio.wait_for(observe_progress(), 10)
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == first_output
        history = (await session.history()).items
        assert any(item.item_kind == "tool_result" and item.content == "retained tool result" for item in history)
        terminal_head = await storage.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.tenant_id,
        )
        assert terminal_head is not None
        assert terminal_head.state is (ExecutionHistoryState.SEALED if trace else ExecutionHistoryState.CLOSED)
        assert terminal_head.producer_generation == active_head.producer_generation
        assert terminal_head.producer_claim_id == active_head.producer_claim_id
        seal = await storage.execution.executions.get_history_seal(execution.execution_id, tenant_id=runtime.tenant_id)
        assert (seal is not None) is trace
        assert terminal_head.seal_digest == (seal.seal_digest if seal is not None else None)
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        assert (await archive.get_agent_run(agent_run_id=run_id) is not None) is trace
        assert bool(await archive.list_model_interactions(agent_run_id=run_id)) is trace
        assert bool(await archive.list_events(agent_run_id=run_id)) is trace
        recovery = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        assert (await recovery.latest_checkpoint(agent_run_id=run_id) is not None) is (restart or split_storage)
        if split_storage:
            handoff = await storage.recovery.checkpoints.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert handoff is not None and handoff.state is RecoveryCheckpointState.COMPLETED
            assert handoff.handoff_phase is RecoveryHandoffPhase.COMPLETED
            assert handoff.terminal_handoff is not None

    reopened = storage_factory(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=reopened, capabilities=(group,),
    ) as runtime:
        restored = await runtime.executions.get(execution.execution_id)
        assert (await restored.wait(timeout_seconds=10)).result.output == first_output
        record = await reopened.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None
        assert record.restart_checkpoints is restart and record.durable_trace is trace
        assert await reopened.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) == terminal_head
        archive = reopened.run_store.read_store(RuntimeDomain.EXECUTION)
        assert (await archive.get_agent_run(agent_run_id=run_id) is not None) is trace
        assert bool(await archive.list_model_interactions(agent_run_id=run_id)) is trace
        assert bool(await archive.list_events(agent_run_id=run_id)) is trace
        recovery = reopened.run_store.read_store(RuntimeDomain.RECOVERY)
        assert (await recovery.latest_checkpoint(agent_run_id=run_id) is not None) is (restart or split_storage)
        if trace:
            assert (await restored.trace()).items
            assert (await restored.model_interactions(include_content=True)).items
        session = runtime.agents.get().session("session")
        assert (await session.history()).items == history
        before_replay = list(model_calls)
        replay = await session.start("first question", output=_Answer, idempotency_key="first")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=10)).result.output == first_output
        assert model_calls == before_replay
        second = await session.run(
            "second question", output=_Answer, idempotency_key="second", timeout_seconds=10,
        )
        assert second.result.status is ExecutionStatus.SUCCEEDED
        assert second.result.output == second_output
    assert tool_calls == ["lookup"]
    assert model_calls == ["first question", "first question", "second question"]


@pytest.mark.parametrize("persisted", (False, True), ids=("domain", "persisted"))
def test_execution_wire_requires_admitted_persistence_flags(persisted: bool) -> None:
    record = _execution()
    assert not record.restart_checkpoints and not record.durable_trace
    encode = _encode_persisted_domain if persisted else encode_domain
    payload = encode(record)

    def decode(value: JsonValue) -> ExecutionRecord:
        if persisted:
            return _decode_enveloped_domain(
                encode_envelope({"type": "execution_record", "payload": value}), ExecutionRecord,
            )
        return decode_domain(value, ExecutionRecord)

    assert decode(payload) == record
    for omitted in (("restart_checkpoints",), ("durable_trace",), ("restart_checkpoints", "durable_trace")):
        fields = {name: value for name, value in payload["fields"].items() if name not in omitted}
        with pytest.raises(AIError) as missing:
            decode({**payload, "fields": fields})
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_task_admission_retains_its_explicit_persistence_guarantees() -> None:
    from linktools.ai.core import TaskStatus
    from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
    from ._runtime_test_helpers import RuntimeUsageModels

    async def answer(_context: TaskNodeContext[None]) -> dict[str, int]:
        return {"value": 42}

    task = Task("answer", answer, effect_policy="none")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open(_NAMESPACE, models=RuntimeUsageModels(), storage=storage) as runtime:
        graph = await runtime.tasks.bind(task).start(
            TaskGraph("task-guarantees", (TaskNode("answer", task=task),)),
            idempotency_key="task-guarantees",
        )
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        execution = await graph.execution("answer")
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.binding_kind == "task"
        assert record.restart_checkpoints and record.durable_trace
        assert await graph.result("answer") == {"value": 42}


@pytest.mark.asyncio
@pytest.mark.parametrize("capability_type", (RestartCheckpointCapability, DurableTraceCapability))
async def test_persistence_capabilities_reject_agent_scope(
    capability_type: type[RestartCheckpointCapability | DurableTraceCapability],
) -> None:
    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.capability(capability_type())
    group.agent("default", model="default", allow_capabilities=())

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        raise AssertionError("invalid capability scope must fail before provider execution")

    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            _NAMESPACE, models=_Models(_UsageFunctionModel(model)),
            storage=RuntimeStorage.in_memory(), capabilities=(group,),
        ):
            raise AssertionError("Agent-scoped persistence capability was accepted")
    assert invalid.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen_with_restart", (False, True), ids=("default-reopen", "restart-reopen"))
@pytest.mark.parametrize("trace", (False, True), ids=("no-trace", "trace"))
async def test_recovery_cannot_enable_restart_after_execution_admission(
    tmp_path: Path, reopen_with_restart: bool, trace: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("provider")
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("original producer finished")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=())
    if trace:
        group.capability(DurableTraceCapability(), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "recovery.db"
    original = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        execution = await owner.agents.get().start("hold the original request", idempotency_key="original")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            before = await original.execution.executions.get(execution.execution_id, tenant_id=owner.tenant_id)
            head = await original.execution.executions.get_history_head(execution.execution_id, tenant_id=owner.tenant_id)
            assert before is not None and not before.restart_checkpoints
            assert before.agent_run_seq > 0
            if reopen_with_restart:
                group.capability(RestartCheckpointCapability(), scope="runtime")
            async with Runtime.open(
                _NAMESPACE, models=models, storage=RuntimeStorage.sqlite(database), capabilities=(group,),
            ) as reopened:
                remote = await reopened.executions.get(execution.execution_id)
                replayed = await reopened.agents.get().start(
                    "hold the original request", idempotency_key="original",
                )
                assert replayed.execution_id == execution.execution_id
                assert not reopened._execution_service.runtime_backend().worker_installed(execution.execution_id)
                assert calls == ["provider"]
                with pytest.raises(AIError) as unavailable:
                    await remote.recover(idempotency_key="unsupported-restart")
                assert unavailable.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
                assert not unavailable.value.retryable
                assert calls == ["provider"]
                assert await original.execution.executions.get(
                    execution.execution_id, tenant_id=owner.tenant_id,
                ) == before
                current_head = await original.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=owner.tenant_id,
                )
                assert current_head is not None and head is not None
                assert current_head.producer_generation == head.producer_generation
                assert current_head.producer_claim_id == head.producer_claim_id
        finally:
            release.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert calls == ["provider"]


@pytest.mark.asyncio
@pytest.mark.parametrize("late_response", ("text", "tool"))
@pytest.mark.parametrize("trace", (False, True), ids=("no-trace", "trace"))
async def test_remote_force_cancel_releases_session_and_fences_late_provider(
    tmp_path: Path, late_response: str, trace: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []
    effects: list[str] = []

    async def late_effect(_context: AgentContext[None]) -> str:
        effects.append("effect")
        return "effect result"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        prompt = next(part.content for message in reversed(messages) for part in message.parts
                      if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        calls.append(prompt)
        if prompt == "hold the provider":
            entered.set()
            await release.wait()
            if late_response == "tool":
                return ModelResponse(parts=[ToolCallPart("late_effect", {}, tool_call_id="late-call")])
            return ModelResponse(parts=[TextPart("late answer")])
        return ModelResponse(parts=[TextPart("next turn finished")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.tool(late_effect, effect_policy="non_replay_safe")
    group.agent("default", model="default", allow_tools=("late_effect",))
    if trace:
        group.capability(DurableTraceCapability(), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "cancel.db"
    original = RuntimeStorage.sqlite(database)
    remote_state = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        await owner.agents.get().create_session("session")
        execution = await owner.agents.get().session("session").start("hold the provider", idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            # Public wait can return the remote terminal result before this old
            # producer exits; retain its task to verify the later write boundary.
            worker = owner._execution_service.runtime_backend()._tasks[execution.execution_id]
            async with Runtime.open(
                _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
            ) as observer:
                remote = await observer.executions.get(execution.execution_id)
                normal = await remote.cancel(idempotency_key="normal-cancel")
                assert not normal.cancelled
                cancelling = await remote_state.execution.executions.get(
                    execution.execution_id, tenant_id=observer.tenant_id,
                )
                assert cancelling is not None and cancelling.status is ExecutionStatus.CANCELLING
                assert not worker.done()
                with pytest.raises(AIError) as busy:
                    await observer.agents.get().session("session").start("still busy", idempotency_key="busy")
                assert busy.value.code is ErrorCode.SESSION_BUSY

                forced = await remote.cancel(idempotency_key="force-cancel", force=True)
                assert forced.cancelled
                terminal = await remote_state.execution.executions.get(
                    execution.execution_id, tenant_id=observer.tenant_id,
                )
                assert terminal is not None and terminal.status is ExecutionStatus.CANCELLED
                head = await remote_state.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=observer.tenant_id,
                )
                assert head is not None
                assert head.state is (ExecutionHistoryState.SEALED if trace else ExecutionHistoryState.CLOSED)
                session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                assert session is not None and session.active_execution_id is None
                assert (await remote.cancel(idempotency_key="force-cancel", force=True)).cancelled
                assert calls == ["hold the provider"] and effects == []

                release.set()
                await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
                assert await remote_state.execution.executions.get(
                    execution.execution_id, tenant_id=observer.tenant_id,
                ) == terminal
                assert await remote_state.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=observer.tenant_id,
                ) == head
                assert (await remote.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                assert effects == []
                next_turn = await observer.agents.get().session("session").run(
                    "next question", idempotency_key="next", timeout_seconds=10,
                )
                assert next_turn.result.status is ExecutionStatus.SUCCEEDED
                assert next_turn.result.output == {"text": "next turn finished"}
        finally:
            release.set()
    assert calls == ["hold the provider", "next question"]
    assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed", (False, True), ids=("before-claim", "after-claim"))
async def test_remote_force_cancel_orders_with_tool_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claimed: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.runtime.state._contracts import ToolOperationAdmission, ToolOperationRecord
    from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands

    reached = asyncio.Event()
    release = asyncio.Event()
    effects: list[str] = []
    original_admit = RuntimeStateCommands.commit_tool_admission

    async def pause_admission(
        commands: RuntimeStateCommands, request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        record = await original_admit(commands, request) if claimed else None
        reached.set()
        await release.wait()
        return record if record is not None else await original_admit(commands, request)

    monkeypatch.setattr(RuntimeStateCommands, "commit_tool_admission", pause_admission)

    async def effect(_context: AgentContext[None]) -> str:
        effects.append("effect")
        return "confirmed result"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("finished")])
        return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="effect-call")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.tool(effect, effect_policy="non_replay_safe")
    group.agent("default", model="default", allow_tools=("effect",))
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "admission-race.db"
    original = RuntimeStorage.sqlite(database)
    remote_state = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        await owner.agents.get().create_session("session")
        execution = await owner.agents.get().session("session").start("perform effect")
        try:
            await asyncio.wait_for(reached.wait(), 10)
            worker = owner._execution_service.runtime_backend()._tasks[execution.execution_id]
            async with Runtime.open(
                _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
            ) as observer:
                remote = await observer.executions.get(execution.execution_id)
                operations = await remote_state.recovery.tools.list_by_execution(
                    execution.execution_id, tenant_id=observer.tenant_id,
                )
                assert [operation.status for operation in operations] == (
                    [ToolOperationStatus.CLAIMED] if claimed else []
                )
                outcome = await remote.cancel(idempotency_key="force-at-claim", force=True)
                assert outcome.cancelled is (not claimed)
                record = await remote_state.execution.executions.get(
                    execution.execution_id, tenant_id=observer.tenant_id,
                )
                assert record is not None
                assert record.status is (ExecutionStatus.CANCELLING if claimed else ExecutionStatus.CANCELLED)
                session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                assert session is not None
                assert session.active_execution_id == (execution.execution_id if claimed else None)
                assert effects == []
                release.set()
                await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
                if not claimed:
                    assert effects == []
                    assert await remote_state.recovery.tools.list_by_execution(
                        execution.execution_id, tenant_id=observer.tenant_id,
                    ) == ()
                    assert await remote_state.execution.executions.get(
                        execution.execution_id, tenant_id=observer.tenant_id,
                    ) == record
                else:
                    assert (await remote.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution_kind", ("applied", "not_applied"))
async def test_remote_force_cancel_requires_explicit_tool_effect_resolution(
    tmp_path: Path, resolution_kind: str,
) -> None:
    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.runtime import ToolEffectApplied, ToolEffectNotApplied

    effects: list[str] = []
    requests: list[str] = []

    async def uncertain_effect(_context: AgentContext[None]) -> str:
        effects.append("effect started")
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        requests.append("request")
        return ModelResponse(parts=[ToolCallPart("uncertain_effect", {}, tool_call_id="uncertain-call")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.tool(uncertain_effect, effect_policy="non_replay_safe")
    group.agent("default", model="default", allow_tools=("uncertain_effect",))
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "unknown-effect.db"
    original = RuntimeStorage.sqlite(database)
    remote_state = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        await owner.agents.get().create_session("session")
        execution = await owner.agents.get().session("session").start("perform uncertain effect")
        with pytest.raises(AIError) as unknown:
            await execution.wait(timeout_seconds=10)
        assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        async with Runtime.open(
            _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
        ) as observer:
            remote = await observer.executions.get(execution.execution_id)
            assert not (await remote.cancel(idempotency_key="force-unknown", force=True)).cancelled
            record = await remote_state.execution.executions.get(execution.execution_id, tenant_id=observer.tenant_id)
            assert record is not None and record.status is ExecutionStatus.RECOVERY_REQUIRED
            assert await remote_state.execution.executions.get_result(
                execution.execution_id, tenant_id=observer.tenant_id,
            ) is None
            operations = await remote_state.recovery.tools.list_by_execution(
                execution.execution_id, tenant_id=observer.tenant_id,
            )
            assert len(operations) == 1 and operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
            assert len(await remote.recovery_effects()) == 1
            session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
            assert session is not None and session.active_execution_id == execution.execution_id
            with pytest.raises(AIError) as busy:
                await observer.agents.get().session("session").start("next question", idempotency_key="busy")
            assert busy.value.code is ErrorCode.SESSION_BUSY
            effect = operations[0]
            resolved = await remote.resolve_tool_effect(
                effect.tool_operation_id, expected_fence=effect.fence,
                resolution=(ToolEffectApplied({"confirmed": True})
                            if resolution_kind == "applied" else ToolEffectNotApplied()),
                idempotency_key="resolve-effect",
            )
            assert resolved.status is (
                ToolOperationStatus.COMPLETED if resolution_kind == "applied" else ToolOperationStatus.PENDING
            )
            assert await remote.recovery_effects() == ()
            assert (await remote.cancel(idempotency_key="force-unknown", force=True)).cancelled
            terminal = await remote_state.execution.executions.get(
                execution.execution_id, tenant_id=observer.tenant_id,
            )
            assert terminal is not None and terminal.status is ExecutionStatus.CANCELLED
            assert (await remote.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
            session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
            assert session is not None and session.active_execution_id is None
            head = await remote_state.execution.executions.get_history_head(
                execution.execution_id, tenant_id=observer.tenant_id,
            )
            assert head is not None and head.state is ExecutionHistoryState.CLOSED
            assert (await remote.cancel(idempotency_key="force-unknown", force=True)).cancelled
    assert effects == ["effect started"]
    assert requests == ["request"]


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_kind", ("restartable", "task-bound"))
async def test_force_cancel_does_not_revoke_nonordinary_execution_ownership(execution_kind: str) -> None:
    from linktools.ai.runtime._execution import CancelEffectOutcome

    from .test_local_execution_recovery import _ToolOperations, _backend, _record

    backend = _backend()
    record = replace(
        _record(), status=ExecutionStatus.CANCELLING, agent_run_seq=1,
        restart_checkpoints=execution_kind == "restartable", durable_trace=False,
        requires_task_invocation_capture=execution_kind == "task-bound",
    )
    backend._execution.executions.record = record
    backend._tool_operations = _ToolOperations(())
    assert await backend.cancel(record, force=True) is CancelEffectOutcome.UNKNOWN
    assert backend._execution.executions.record == record


@pytest.mark.asyncio
@pytest.mark.parametrize("root_state,depth", (
    ("missing", 2), ("unreadable", 2), ("not-root", 2), ("task-bound", 1), ("task-bound", 2),
))
async def test_delegated_force_cancel_requires_available_ordinary_root(
    monkeypatch: pytest.MonkeyPatch, root_state: str, depth: int,
) -> None:
    from linktools.ai.core import ExecutionLineageKind
    from linktools.ai.runtime._execution import CancelEffectOutcome

    from .test_local_execution_recovery import _ToolOperations, _backend, _record

    backend = _backend()
    root = replace(
        _record(), execution_id="root", root_execution_id="root",
        status=ExecutionStatus.CANCELLING, agent_run_seq=1,
        restart_checkpoints=False, durable_trace=False,
        requires_task_invocation_capture=root_state == "task-bound",
    )
    if root_state == "not-root":
        root = replace(root, root_execution_id="another-root")
    descendant = replace(
        root, execution_id="descendant", root_execution_id="root",
        parent_execution_id="root" if depth == 1 else "intermediate",
        parent_invocation_id="delegate-call", lineage_kind=ExecutionLineageKind.SUBAGENT,
        requires_task_invocation_capture=False,
    )

    async def get(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        assert tenant_id == "tenant"
        if execution_id == descendant.execution_id:
            return descendant
        assert execution_id == "root"
        if root_state == "unreadable":
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return None if root_state == "missing" else root

    monkeypatch.setattr(backend._execution.executions, "get", get)
    backend._tool_operations = _ToolOperations(())
    if root_state == "task-bound":
        assert await backend.cancel(descendant, force=True) is CancelEffectOutcome.UNKNOWN
    else:
        with pytest.raises(AIError) as unavailable:
            await backend.cancel(descendant, force=True)
        assert unavailable.value.code is (
            ErrorCode.STORAGE_UNAVAILABLE if root_state == "unreadable" else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    assert await get(descendant.execution_id, tenant_id="tenant") == descendant


@pytest.mark.asyncio
async def test_remote_force_cancel_recursively_fences_ordinary_delegated_tree(tmp_path: Path) -> None:
    from pydantic_ai.messages import TextPart

    entered = {prompt: asyncio.Event() for prompt in ("root", "child", "grandchild")}
    release = asyncio.Event()
    calls: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        prompt = next(part.content for message in reversed(messages) for part in message.parts
                      if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        calls.append(prompt)
        entered[prompt].set()
        await release.wait()
        return ModelResponse(parts=[TextPart(f"late {prompt}")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "delegated-cancel.db"
    original = RuntimeStorage.sqlite(database)
    remote_state = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        await owner.agents.get().create_session("session")
        root = await owner.agents.get().session("session").start("root", idempotency_key="root")
        try:
            await asyncio.wait_for(entered["root"].wait(), 10)
            child_id = await _start_delegated_execution(owner, original, root.execution_id, "child")
            await asyncio.wait_for(entered["child"].wait(), 10)
            grandchild_id = await _start_delegated_execution(owner, original, child_id, "grandchild")
            await asyncio.wait_for(entered["grandchild"].wait(), 10)
            execution_ids = (root.execution_id, child_id, grandchild_id)
            workers = tuple(owner._execution_service.runtime_backend()._tasks[key] for key in execution_ids)
            async with Runtime.open(
                _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
            ) as observer:
                assert [item.execution_id for item in await observer.executions.list_children(
                    root.execution_id, principal=observer.default_principal,
                )] == [child_id]
                assert [item.execution_id for item in await observer.executions.list_children(
                    child_id, principal=observer.default_principal,
                )] == [grandchild_id]
                remote = await observer.executions.get(root.execution_id)
                assert (await remote.cancel(idempotency_key="cancel-tree", force=True)).cancelled
                terminal_records = []
                terminal_heads = []
                for execution_id in execution_ids:
                    record = await remote_state.execution.executions.get(execution_id, tenant_id=observer.tenant_id)
                    assert record is not None and record.status is ExecutionStatus.CANCELLED
                    assert not record.restart_checkpoints and not record.durable_trace
                    assert record.root_execution_id == root.execution_id
                    terminal_records.append(record)
                    head = await remote_state.execution.executions.get_history_head(
                        execution_id, tenant_id=observer.tenant_id,
                    )
                    assert head is not None and head.state is ExecutionHistoryState.CLOSED
                    terminal_heads.append(head)
                session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                assert session is not None and session.active_execution_id is None
                assert (await remote.cancel(idempotency_key="cancel-tree", force=True)).cancelled
                assert all(not worker.done() for worker in workers)
                release.set()
                await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), 10)
                for execution_id, record, head in zip(execution_ids, terminal_records, terminal_heads):
                    assert await remote_state.execution.executions.get(execution_id, tenant_id=observer.tenant_id) == record
                    assert await remote_state.execution.executions.get_history_head(
                        execution_id, tenant_id=observer.tenant_id,
                    ) == head
        finally:
            release.set()
    assert calls == ["root", "child", "grandchild"]


@pytest.mark.asyncio
async def test_unknown_delegated_tool_effect_prevents_parent_cancel_and_session_release(tmp_path: Path) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.core import ToolOperationStatus
    from linktools.ai.runtime import ToolEffectApplied

    entered = asyncio.Event()
    release = asyncio.Event()
    requests: list[str] = []
    effects: list[str] = []

    async def uncertain_effect(_context: AgentContext[None]) -> str:
        effects.append("uncertain child effect")
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        prompt = next(part.content for message in reversed(messages) for part in message.parts
                      if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        requests.append(prompt)
        if prompt == "child":
            return ModelResponse(parts=[ToolCallPart("uncertain_effect", {}, tool_call_id="child-effect")])
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("late parent answer")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.tool(uncertain_effect, effect_policy="non_replay_safe")
    group.agent("default", model="default", allow_tools=("uncertain_effect",))
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "delegated-effect.db"
    original = RuntimeStorage.sqlite(database)
    remote_state = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        await owner.agents.get().create_session("session")
        root = await owner.agents.get().session("session").start("root", idempotency_key="root")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            root_worker = owner._execution_service.runtime_backend()._tasks[root.execution_id]
            child_id = await _start_delegated_execution(owner, original, root.execution_id, "child")
            child = await owner.executions.get(child_id)
            with pytest.raises(AIError) as unknown:
                await child.wait(timeout_seconds=10)
            assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            async with Runtime.open(
                _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
            ) as observer:
                remote = await observer.executions.get(root.execution_id)
                with pytest.raises(AIError) as unresolved:
                    await remote.cancel(idempotency_key="cancel-unknown-tree", force=True)
                assert unresolved.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                parent_record = await remote_state.execution.executions.get(root.execution_id, tenant_id=observer.tenant_id)
                assert parent_record is not None and parent_record.status is ExecutionStatus.CANCELLING
                child_record = await remote_state.execution.executions.get(child_id, tenant_id=observer.tenant_id)
                assert child_record is not None and child_record.status is ExecutionStatus.RECOVERY_REQUIRED
                for execution_id in (root.execution_id, child_id):
                    assert await remote_state.execution.executions.get_result(execution_id, tenant_id=observer.tenant_id) is None
                    head = await remote_state.execution.executions.get_history_head(execution_id, tenant_id=observer.tenant_id)
                    assert head is not None and head.state is ExecutionHistoryState.OPEN
                session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                assert session is not None and session.active_execution_id == root.execution_id
                with pytest.raises(AIError) as busy:
                    await observer.agents.get().session("session").start("next", idempotency_key="busy")
                assert busy.value.code is ErrorCode.SESSION_BUSY
                operations = await remote_state.recovery.tools.list_by_execution(child_id, tenant_id=observer.tenant_id)
                assert len(operations) == 1 and operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
                remote_child = await observer.executions.get(child_id)
                effect = operations[0]
                await remote_child.resolve_tool_effect(
                    effect.tool_operation_id, expected_fence=effect.fence,
                    resolution=ToolEffectApplied({"confirmed": True}), idempotency_key="resolve-child",
                )
                assert (await remote.cancel(idempotency_key="cancel-resolved-tree", force=True)).cancelled
                assert (await remote.cancel(idempotency_key="cancel-resolved-tree", force=True)).cancelled
                session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                assert session is not None and session.active_execution_id is None
                assert (await remote_child.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                release.set()
                await asyncio.wait_for(asyncio.gather(root_worker, return_exceptions=True), 10)
                assert (await remote.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
        finally:
            release.set()
    assert requests == ["root", "child"]
    assert effects == ["uncertain child effect"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task_owned", (False, True), ids=("ordinary", "task-owned"))
@pytest.mark.parametrize("restart", (False, True), ids=("no-restart", "restart"))
@pytest.mark.parametrize("cache_enabled", (False, True), ids=("source", "cache"))
async def test_remote_force_cancel_of_model_delegation_respects_root_owner(
    tmp_path: Path, task_owned: bool, restart: bool, cache_enabled: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.runtime import AgentTaskInput
    from linktools.ai.task import TaskGraph, TaskNode
    from linktools.ai.storage import InMemoryCacheBackend

    from ._runtime_test_helpers import _wait_for_committed

    entered = asyncio.Event()
    release = asyncio.Event()
    requests: list[str] = []
    post_cancel_requests: list[tuple[ExecutionStatus, ExecutionStatus]] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = next(part.content for message in reversed(messages) for part in message.parts
                      if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        requests.append(prompt)
        if prompt == "root":
            if requests.count("root") > 1:
                current_root = await original.execution.executions.get(root_id, tenant_id=owner.tenant_id)
                current_child = await original.execution.executions.get(child_id, tenant_id=owner.tenant_id)
                assert current_root is not None and current_child is not None
                post_cancel_requests.append((current_root.status, current_child.status))
                return ModelResponse(parts=[TextPart("late parent answer")])
            assert "delegate_task" in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[ToolCallPart(
                "delegate_task", {"subagent_id": "child", "task": "child"}, tool_call_id="delegate-child",
            )])
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("late child answer")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",))
    group.agent("child", model="default", allow_tools=(), allow_subagents=())
    if restart:
        group.capability(RestartCheckpointCapability(), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "model-delegation.db"
    original = RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1024 * 1024) if cache_enabled else None)
    remote_state = RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1024 * 1024) if cache_enabled else None)
    async with Runtime.open(
        _NAMESPACE, models=models, storage=original, capabilities=(group,),
    ) as owner:
        if task_owned:
            task = owner.tasks.from_agent("delegation.agent", owner.agents.get())
            run = await owner.tasks.bind(task).start(
                TaskGraph("delegation", (TaskNode("agent", task=task, input=AgentTaskInput("root")),)),
                idempotency_key="task-root",
            )
            state = await _wait_for_committed(run.state, lambda value: value.node_states[0].execution_id is not None)
            root_id = state.node_states[0].execution_id
        else:
            await owner.agents.get().create_session("session")
            root = await owner.agents.get().session("session").start("root", idempotency_key="root")
            root_id = root.execution_id
        assert root_id is not None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            child_view, = await owner.executions.list_children(root_id, principal=owner.default_principal)
            child_id = child_view.execution_id
            root_record = await original.execution.executions.get(root_id, tenant_id=owner.tenant_id)
            child_record = await original.execution.executions.get(child_id, tenant_id=owner.tenant_id)
            assert root_record is not None and root_record.requires_task_invocation_capture is task_owned
            assert child_record is not None and not child_record.requires_task_invocation_capture
            assert child_record.root_execution_id == root_id
            workers = tuple(owner._execution_service.runtime_backend()._tasks[key] for key in (root_id, child_id))
            async with Runtime.open(
                _NAMESPACE, models=models, storage=remote_state, capabilities=(group,),
            ) as observer:
                remote = await observer.executions.get(root_id)
                remote_child = await observer.executions.get(child_id)
                if task_owned or restart:
                    assert not (await remote_child.cancel(idempotency_key="child-force", force=True)).cancelled
                    with pytest.raises(AIError) as guarded:
                        await remote.cancel(idempotency_key="root-force", force=True)
                    assert guarded.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                    for execution_id in (root_id, child_id):
                        current = await remote_state.execution.executions.get(execution_id, tenant_id=observer.tenant_id)
                        assert current is not None and current.status is ExecutionStatus.CANCELLING
                        assert await remote_state.execution.executions.get_result(execution_id, tenant_id=observer.tenant_id) is None
                        head = await remote_state.execution.executions.get_history_head(execution_id, tenant_id=observer.tenant_id)
                        assert head is not None and head.state is ExecutionHistoryState.OPEN
                    assert all(not worker.done() for worker in workers)
                else:
                    assert (await remote.cancel(idempotency_key="root-force", force=True)).cancelled
                    assert (await remote.cancel(idempotency_key="root-force", force=True)).cancelled
                    assert (await remote_child.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                    session = await remote_state.conversation.sessions.get("session", tenant_id=observer.tenant_id)
                    assert session is not None and session.active_execution_id is None
                release.set()
                await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), 10)
                assert (await remote.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                assert (await remote_child.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                if task_owned or restart:
                    await observer._execution_service.runtime_backend().cancel_children(root_id, observer.default_principal)
                if task_owned:
                    await run.wait(timeout_seconds=10)
        finally:
            release.set()
    assert requests == ["root", "child"], post_cancel_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True), ids=("no-restart", "restart"))
@pytest.mark.parametrize("split_storage", (False, True), ids=("combined-storage", "split-storage"))
async def test_no_trace_terminal_readback_preserves_session_and_single_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart: bool, split_storage: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.observe import Metrics, Observation
    from linktools.ai.observe._memory import InMemoryMetricStore
    from linktools.ai.runtime.state._contracts import ExecutionTerminalCommit, ExecutionTerminalCommitResult
    from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands

    original_commit = RuntimeStateCommands.commit_terminal_checkpoint
    committed_unknown = False
    terminal_metrics: list[Observation] = []

    async def commit(
        commands: RuntimeStateCommands, request: ExecutionTerminalCommit, **kwargs: object,
    ) -> ExecutionTerminalCommitResult:
        nonlocal committed_unknown
        result = await original_commit(commands, request, **kwargs)
        if not committed_unknown:
            committed_unknown = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        return result

    monkeypatch.setattr(RuntimeStateCommands, "commit_terminal_checkpoint", commit)

    class MetricStore(InMemoryMetricStore):
        async def put_observations(self, namespace: str, observations: tuple[Observation, ...]) -> None:
            terminal_metrics.extend(item for item in observations if item.kind == "linktools.execution.terminal")
            await super().put_observations(namespace, observations)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart("durable answer")])

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=())
    if restart:
        group.capability(RestartCheckpointCapability(), scope="runtime")
    storage_factory = _split_sqlite_storage if split_storage else RuntimeStorage.sqlite
    storage = storage_factory(tmp_path / "terminal.db")
    async with Runtime.open(
        _NAMESPACE, models=_Models(_UsageFunctionModel(model)), storage=storage, capabilities=(group,),
        metrics=Metrics.from_store(MetricStore(), namespace=_NAMESPACE),
    ) as runtime:
        await runtime.agents.get().create_session("session")
        session = runtime.agents.get().session("session")
        execution = await session.start("question", idempotency_key="question")
        result = (await execution.wait(timeout_seconds=10)).result
        assert committed_unknown
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": "durable answer"}
        history = (await session.history()).items
        assert [item.content for item in history if item.item_kind == "user"] == ["question"]
        assert [item.content for item in history if item.item_kind == "assistant"] == ["durable answer"]
        stored_session = await storage.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
        assert stored_session is not None and stored_session.active_execution_id is None
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.status is ExecutionStatus.SUCCEEDED
        assert record.restart_checkpoints is restart and not record.durable_trace
        head = await storage.execution.executions.get_history_head(execution.execution_id, tenant_id=runtime.tenant_id)
        assert head is not None and head.state is ExecutionHistoryState.CLOSED
        assert await storage.execution.executions.get_history_seal(execution.execution_id, tenant_id=runtime.tenant_id) is None
        run_id = agent_run_id(
            namespace=_NAMESPACE, tenant_id=runtime.tenant_id,
            execution_id=execution.execution_id, agent_run_seq=record.agent_run_seq,
        )
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        assert await archive.get_agent_run(agent_run_id=run_id) is None
        assert not await archive.list_events(agent_run_id=run_id)
        assert not await archive.list_model_interactions(agent_run_id=run_id)
        replay = await session.start("question", idempotency_key="question")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait(timeout_seconds=10)).result == result
        assert (await session.history()).items == history
    assert len(terminal_metrics) == 1
    assert terminal_metrics[0].status == "SUCCEEDED"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("reopen_with_restart", (False, True), ids=("default-reopen", "restart-reopen"))
async def test_default_admission_without_worker_never_restarts_on_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, reopen_with_restart: bool,
) -> None:
    from pydantic_ai.messages import TextPart

    from linktools.ai.runtime.service_api import ExecutionRequest

    requests: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        requests.append("unexpected restart")
        return ModelResponse(parts=[TextPart("unexpected restart")])

    def storage() -> RuntimeStorage:
        if backend == "filesystem":
            return RuntimeStorage.filesystem(tmp_path / "runtime")
        return RuntimeStorage.sqlite(tmp_path / "runtime.db")

    group: CapabilityGroup[None] = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    initial = storage()
    async with Runtime.open(
        _NAMESPACE, models=models, storage=initial, capabilities=(group,),
    ) as runtime:
        await runtime.agents.get().create_session("session")

        async def interrupted_launch(
            request: ExecutionRequest, execution: ExecutionRecord, **kwargs: object,
        ) -> None:
            del request, execution, kwargs

        with monkeypatch.context() as fault:
            fault.setattr(runtime._execution_service.runtime_backend(), "launch", interrupted_launch)
            execution = await runtime.agents.get().session("session").start(
                "accepted without a worker", idempotency_key="admission",
            )
        admitted = await initial.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert admitted is not None and admitted.status is ExecutionStatus.STARTED
        assert admitted.agent_run_seq == 0
        assert not admitted.restart_checkpoints and not admitted.durable_trace
        head = await initial.execution.executions.get_history_head(execution.execution_id, tenant_id=runtime.tenant_id)
        assert head is not None
        assert requests == []

    if reopen_with_restart:
        group.capability(RestartCheckpointCapability(), scope="runtime")
    reopened = storage()
    async with Runtime.open(
        _NAMESPACE, models=models, storage=reopened, capabilities=(group,),
    ) as runtime:
        assert await reopened.execution.executions.get(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) == admitted
        assert await reopened.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) == head
        assert not runtime._execution_service.runtime_backend().worker_installed(execution.execution_id)
        assert requests == []
        restored = await runtime.executions.get(execution.execution_id)
        with pytest.raises(AIError) as unavailable:
            await restored.recover(idempotency_key="unsupported-recovery")
        assert unavailable.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        assert not unavailable.value.retryable
        assert not runtime._execution_service.runtime_backend().worker_installed(execution.execution_id)
        assert await reopened.execution.executions.get(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) == admitted
        assert await reopened.execution.executions.get_history_head(
            execution.execution_id, tenant_id=runtime.tenant_id,
        ) == head
        session = await reopened.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
        assert session is not None and session.active_execution_id == execution.execution_id
        assert requests == []
