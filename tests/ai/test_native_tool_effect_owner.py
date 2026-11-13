#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native tool observations and runtime dispatch share one effect authority."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.step_persistence import (
    InMemoryStepStore, RunRecord, StepPersistence, ToolEffectRecord, annotate_tool_effect,
)

from linktools.ai.core import ExecutionStatus, Principal, ToolOperationStatus, agent_conversation_id, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.runs import AgentRunRecord
from linktools.ai.history import History
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from linktools.ai.storage import InMemoryObjectStore

from ._runtime_test_helpers import tool_run_context, tool_with_metadata
from .test_history_projection_conformance import _record


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def state(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Persistence]:
    storage = (PersistenceStorage.in_memory() if request.param == "memory"
               else PersistenceStorage.sqlite(tmp_path / "effects.db"))
    persistence = Persistence(storage)
    await persistence.initialize(namespace="native-effects", tenant_id="tenant")
    try:
        yield persistence
    finally:
        await persistence.close()


def _bridge(state: Persistence, *, run_id: str = "run", source_run_id: str | None = None,
            binding_digest: str = "b" * 64, operation_ids: dict[str, str] | None = None) -> RuntimeToolOperationBridge:
    if operation_ids is None and source_run_id is not None:
        from linktools.ai.execution._tool_identity import tool_operation_id
        operation_ids = {"call": tool_operation_id(tenant_id="tenant", execution_id="execution",
                                                  agent_run_id=source_run_id, tool_call_id="call")}
    return RuntimeToolOperationBridge(
        state.checkpoint.tools, InMemoryObjectStore(), namespace="native-effects",
        tenant_id="tenant", execution_id="execution", agent_run_id=run_id,
        binding_digest=binding_digest, owner="worker", background_tasks=set(),
         recovery_agent_run_id=source_run_id, operation_ids=operation_ids,
    )



@pytest.mark.asyncio
async def test_native_observation_is_not_business_admission(state: Persistence) -> None:
    bridge = _bridge(state)
    effect = ToolEffectRecord(run_id="run", tool_call_id="call", tool_name="inspect", status="started")
    store = RuntimeStepStore(InMemoryStepStore(), bridge)
    await store.record_tool_effect(effect)
    assert await store.get_tool_effect(run_id="run", tool_call_id="call") is None
    assert await store.list_unresolved_tool_effects(run_id="run") == []
    repository = state.checkpoint.tools
    assert not await repository.has_by_agent_run("run", tenant_id="tenant")
    assert await repository.get_operation(bridge._operation_id("run", "call"), tenant_id="tenant") is None
    assert await repository.get_operations((bridge._operation_id("run", "call"),), tenant_id="tenant") == ()
    assert await repository.list_by_execution("execution", tenant_id="tenant") == ()
    assert await bridge.list_operations() == ()
    await store.record_tool_effect(replace(effect, status="completed", ended_at=effect.started_at))
    assert await store.get_tool_effect(run_id="run", tool_call_id="call") is None
    with pytest.raises(RuntimeError, match="admitted effect"):
        await store.record_tool_effect(replace(effect, effect_summary="No authorized call yet"))
    assert await bridge.list_operations() == ()


@pytest.mark.asyncio
async def test_official_hooks_preserve_annotations_and_replay_one_result(state: Persistence) -> None:
    effects: list[int] = []
    returns: list[object] = []
    descriptor = ManagedToolDescriptor("tool_operation", "non_replay_safe", "business")

    for run_id, source_run_id in (("run", None), ("resumed", "run"), ("resumed-again", "resumed")):
        bridge = _bridge(state, run_id=run_id, source_run_id="run" if source_run_id else None)
        underlying = InMemoryStepStore()
        store = RuntimeStepStore(underlying, bridge)

        async def create(ctx: RunContext[None], count: int) -> dict[str, object]:
            observed = await store.get_tool_effect(run_id=run_id, tool_call_id=ctx.tool_call_id)
            assert observed is not None and observed.status == "started"
            assert len(await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")) == 1
            await annotate_tool_effect(store, ctx, idempotency_key="provider-42", effect_summary="Created item 42")
            effects.append(count)
            return {"item": 42, "count": count, "nested": [False, None, {"value": "original"}]}

        def respond(messages, info):
            results = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if not results:
                return ModelResponse(parts=[ToolCallPart("create", {"count": 3}, tool_call_id="call")])
            returns.append(results[-1].content)
            return ModelResponse(parts=[TextPart("done")])

        tools = BoundaryToolset(
            (FunctionToolset([tool_with_metadata(create, descriptor)]),), {"create": descriptor},
            id="native-effects", tool_operations=bridge,
        )
        agent = Agent(FunctionModel(respond), toolsets=[tools], capabilities=[StepPersistence(store=store, run_id=run_id)])
        assert (await agent.run("Create the item")).output == "done"
        observed = await store.get_tool_effect(run_id=run_id, tool_call_id="call")
        assert observed is not None and observed.run_id == run_id and observed.status == "completed"
        assert observed.idempotency_key == "provider-42" and observed.effect_summary == "Created item 42"
        assert await store.list_unresolved_tool_effects(run_id=run_id) == []
        assert await underlying.get_tool_effect(run_id=run_id, tool_call_id="call") is None

    assert effects == [3]
    assert returns[0] == returns[1] == returns[2] == {"item": 42, "count": 3, "nested": [False, None, {"value": "original"}]}
    operations = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
    assert len(operations) == 1
    assert operations[0].agent_run_id == "run" and operations[0].status is ToolOperationStatus.COMPLETED
    assert (await state.checkpoint.tools.get_operations((operations[0].tool_operation_id,), tenant_id="tenant"))[0] == operations[0]


@pytest.mark.asyncio
async def test_native_error_hook_cannot_resolve_unknown_external_effect(state: Persistence) -> None:
    bridge = _bridge(state)
    underlying = InMemoryStepStore()
    store = RuntimeStepStore(underlying, bridge)
    descriptor = ManagedToolDescriptor("tool_operation", "non_replay_safe", "business")
    effects: list[str] = []

    async def create() -> str:
        effects.append("dispatched")
        raise RuntimeError("connection lost after dispatch")

    tools = BoundaryToolset(
        (FunctionToolset([tool_with_metadata(create, descriptor)]),), {"create": descriptor},
        id="native-effects", tool_operations=bridge,
    )
    agent = Agent(FunctionModel(lambda messages, info: ModelResponse(
        parts=[ToolCallPart("create", {}, tool_call_id="call")],
    )), toolsets=[tools], capabilities=[StepPersistence(store=store, run_id="run")])
    with pytest.raises(AIError) as failure:
        await agent.run("Create the item")
    assert failure.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    effect = await store.get_tool_effect(run_id="run", tool_call_id="call")
    assert effect is not None and effect.status == "started" and effect.ended_at is None
    assert effect.effect_summary is None
    assert await store.list_unresolved_tool_effects(run_id="run") == [effect]
    assert any(event.kind == "tool_call_failed" for event in await store.list_events(run_id="run"))
    resumed = _bridge(state, run_id="resumed", source_run_id="run")
    await RuntimeStepStore(InMemoryStepStore(), resumed).record_tool_effect(ToolEffectRecord(run_id="resumed", tool_call_id="call", tool_name="create", status="started"))
    with pytest.raises(AIError) as recovery:
        await resumed.begin(tool_run_context(), ToolCallPart("create", {}, tool_call_id="call"), ToolDefinition(name="create"), {}, False)
    assert recovery.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert effects == ["dispatched"]
    assert len(await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")) == 1
    assert (await resumed.list_unresolved_tool_effects(run_id="resumed"))[0].run_id == "resumed"


@pytest.mark.asyncio
async def test_native_observation_keeps_admission_argument_and_binding_checks(state: Persistence) -> None:
    bridge = _bridge(state)
    await RuntimeStepStore(InMemoryStepStore(), bridge).record_tool_effect(ToolEffectRecord(run_id="run", tool_call_id="call", tool_name="create", status="started"))
    call = ToolCallPart("create", {"count": 3}, tool_call_id="call")
    decision = await bridge.begin(tool_run_context(), call, ToolDefinition(name="create"), {"count": 3}, False)
    assert decision is not None
    await bridge.complete(decision, "original")
    for candidate, args, replay_safe in (
        (_bridge(state, run_id="resumed", source_run_id="run"), {"count": 4}, False),
        (_bridge(state, run_id="resumed", source_run_id="run", binding_digest="c" * 64), {"count": 3}, False),
    ):
        with pytest.raises(AIError) as conflict:
            await candidate.begin(tool_run_context(), call, ToolDefinition(name="create"), args, replay_safe)
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT

    replayed = await _bridge(state, run_id="resumed", source_run_id="run").begin(
        tool_run_context(), call, ToolDefinition(name="create"), {"count": 3}, True,
    )
    assert replayed.has_cached_result and replayed.cached_result == "original"
    assert replayed.replay_safe is False


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_cold_recovery_operation_reference_keeps_original_effect_owner(tmp_path: Path, unknown: bool) -> None:
    database = tmp_path / "cold-effects.db"
    call = ToolCallPart("create", {"count": 3}, tool_call_id="call")
    original_operation = None
    for run_id, source_run_id in (("run", None), ("resumed", "run"), ("resumed-again", "resumed")):
        state = Persistence(PersistenceStorage.sqlite(database))
        await state.initialize(namespace="native-effects", tenant_id="tenant")
        try:
            bridge = _bridge(state, run_id=run_id, source_run_id=source_run_id,
                             operation_ids=None if original_operation is None else {"call": original_operation})
            if source_run_id is not None:
                effect = await bridge.get_tool_effect(run_id=run_id, tool_call_id="call")
                assert effect is not None and effect.run_id == run_id
                assert effect.status == ("started" if unknown else "completed")
                unresolved = await bridge.list_unresolved_tool_effects(run_id=run_id)
                assert len(unresolved) == int(unknown)
            await RuntimeStepStore(InMemoryStepStore(), bridge).record_tool_effect(ToolEffectRecord(
                run_id=run_id, tool_call_id="call", tool_name="create", status="started",
            ))
            if unknown and source_run_id is not None:
                with pytest.raises(AIError) as error:
                    await bridge.begin(tool_run_context(), call, ToolDefinition(name="create"), {"count": 3}, False)
                assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            else:
                decision = await bridge.begin(tool_run_context(), call, ToolDefinition(name="create"), {"count": 3}, False)
                assert decision is not None
                if source_run_id is None:
                    original_operation = decision.operation_id
                    if unknown:
                        with pytest.raises(AIError) as error:
                            await bridge.unknown(decision, RuntimeError("dispatch outcome unknown"))
                        assert error.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
                    else:
                        await bridge.complete(decision, {"item": 42})
                else:
                    assert decision.operation_id == original_operation
                    assert decision.has_cached_result and decision.cached_result == {"item": 42}
                    effect = await bridge.get_tool_effect(run_id=run_id, tool_call_id="call")
                    await bridge.record_tool_effect(replace(effect, effect_summary=f"Confirmed in {run_id}"))
                    assert (await bridge.get_tool_effect(run_id=run_id, tool_call_id="call")).effect_summary == f"Confirmed in {run_id}"
            operations = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
            assert len(operations) == 1 and operations[0].agent_run_id == "run"
            assert operations[0].tool_operation_id == original_operation
        finally:
            await state.close()


@pytest.mark.asyncio
async def test_official_terminal_hook_uses_committed_receipt(state: Persistence, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _bridge(state)
    store = RuntimeStepStore(InMemoryStepStore(), bridge)
    capability = StepPersistence(store=store, run_id="run")
    context = tool_run_context()
    call = ToolCallPart("create", {"count": 3}, tool_call_id="call")
    definition = ToolDefinition(name="create")
    await capability.before_tool_execute(context, call=call, tool_def=definition, args={"count": 3})
    decision = await bridge.begin(context, call, definition, {"count": 3}, False)
    assert decision is not None
    await bridge.complete(decision, {"item": 42})

    async def unexpected_storage(*args, **kwargs):
        raise AssertionError("a confirmed same-owner receipt must not be reread or rewritten")

    monkeypatch.setattr(state.checkpoint.tools, "get_operation", unexpected_storage)
    monkeypatch.setattr(state.checkpoint.tools, "record_tool_effect", unexpected_storage)
    result = await capability.after_tool_execute(context, call=call, tool_def=definition,
                                                 args={"count": 3}, result={"item": 42})
    assert result == {"item": 42}
    effect = await store.get_tool_effect(run_id="run", tool_call_id="call")
    assert effect is not None and effect.status == "completed"


@pytest.mark.asyncio
async def test_recovered_effect_queries_keep_canonical_owner_and_exclude_siblings(state: Persistence) -> None:
    original_id, resumed_id, sibling_id = (
        agent_run_id(namespace="native-effects", tenant_id="tenant", execution_id="execution", agent_run_seq=sequence)
        for sequence in (1, 2, 3)
    )
    await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 2))
    call = ToolCallPart("create", {}, tool_call_id="call")
    original = _bridge(state, run_id=original_id)
    decision = await original.begin(tool_run_context(), call, ToolDefinition(name="create"), {}, False)
    assert decision is not None
    with pytest.raises(AIError):
        await original.unknown(decision, RuntimeError("uncertain"))
    sibling = _bridge(state, run_id=sibling_id)
    other = await sibling.begin(tool_run_context(), call, ToolDefinition(name="create"), {}, False)
    assert other is not None and other.operation_id != decision.operation_id
    resumed = _bridge(state, run_id=resumed_id, source_run_id=original_id)
    await RuntimeStepStore(InMemoryStepStore(), resumed).record_tool_effect(ToolEffectRecord(
        run_id=resumed_id, tool_call_id="call", tool_name="create", status="started",
    ))
    with pytest.raises(AIError) as recovery:
        await resumed.begin(tool_run_context(), call, ToolDefinition(name="create"), {}, False)
    assert recovery.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    operations = await resumed.list_operations()
    assert len(operations) == 1
    assert operations[0].agent_run_id == original_id and operations[0].tool_operation_id == decision.operation_id
    recorder = AgentRunRecorder(state.run_store, execution_id="execution", agent_run_id=resumed_id)
    await recorder.register_agent_run(AgentRunRecord(
        resumed_id,
        agent_conversation_id=agent_conversation_id(
            namespace="native-effects", tenant_id="tenant", execution_id="execution",
        ),
        agent_id="default", metadata={"agent_run_seq": "2", "agent_id": "default"},
    ))
    recorder.append_transcript_message(ModelResponse(parts=[call]))
    await recorder.record_event("TOOL_CALL_STARTED", 1, tool_call_id=call.tool_call_id,
                                tool_name=call.tool_name,
                                metadata={"linktools.operation_id": decision.operation_id})
    await recorder.commit_history_boundary()
    await state.run_store.flush_execution_projection(resumed_id, execution_id="execution")
    async with History.open(persistence=state) as history:
        page = await history.history(
            "execution", principal=Principal("reader", "tenant"), agent_run_seq=2,
            tool_call_id="call", include_content=True, limit=10,
        )
    assert len(page.items) == 1
    assert page.items[0].status == "EFFECT_UNKNOWN"
    assert page.items[0].tool_operation_id == decision.operation_id
    assert page.items[0].tool_operation_id != other.operation_id


@pytest.mark.asyncio
async def test_repeated_hot_call_uses_confirmed_terminal_receipt(state: Persistence) -> None:
    bridge = _bridge(state)
    call = ToolCallPart("create", {}, tool_call_id="call")
    definition = ToolDefinition(name="create")
    first = await bridge.begin(tool_run_context(), call, definition, {}, False)
    assert first is not None
    await bridge.complete(first, {"item": 42})
    repeated = await bridge.begin(tool_run_context(), call, definition, {}, False)
    assert repeated is not None and repeated.has_cached_result
    assert repeated.cached_result == {"item": 42}
    with pytest.raises(AIError) as conflict:
        await bridge.begin(tool_run_context(), call, definition, {"changed": True}, False)
    assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("native_observation", [False, True])
async def test_new_recovery_call_belongs_only_to_its_first_observing_attempt(
    state: Persistence, native_observation: bool,
) -> None:
    call = ToolCallPart("create", {}, tool_call_id="new-call")
    definition = ToolDefinition(name="create")
    first = _bridge(state, run_id="resumed", source_run_id="run")
    if native_observation:
        await RuntimeStepStore(InMemoryStepStore(), first).record_tool_effect(ToolEffectRecord(
            run_id="resumed", tool_call_id="new-call", tool_name="create", status="started",
        ))
    decision = await first.begin(tool_run_context(), call, definition, {}, False)
    assert decision is not None
    assert await first.get_tool_effect(run_id="run", tool_call_id="new-call") is None
    assert await first.list_unresolved_tool_effects(run_id="run") == []
    assert len(await first.list_unresolved_tool_effects(run_id="resumed")) == 1
    await first.complete(decision, {"created": True})
    if native_observation:
        prior = await first.get_tool_effect(run_id="resumed", tool_call_id="new-call")
        assert prior is not None
        await first.record_tool_effect(replace(prior, effect_summary="Created in resumed attempt"))
    assert await first.get_tool_effect(run_id="run", tool_call_id="new-call") is None
    assert await state.checkpoint.tools.get_operation(first._operation_id("run", "new-call"), tenant_id="tenant") is None
    assert await _bridge(state).get_tool_effect(run_id="run", tool_call_id="new-call") is None
    assert await _bridge(state).list_unresolved_tool_effects(run_id="run") == []
    for run_id, source_run_id in (("resumed-again", "resumed"), ("last", "resumed-again")):
        resumed = _bridge(state, run_id=run_id, source_run_id=source_run_id,
                          operation_ids={"new-call": decision.operation_id})
        if native_observation:
            await RuntimeStepStore(InMemoryStepStore(), resumed).record_tool_effect(ToolEffectRecord(
                run_id=run_id, tool_call_id="new-call", tool_name="create", status="started",
            ))
        replay = await resumed.begin(tool_run_context(), call, definition, {}, False)
        assert replay is not None and replay.has_cached_result
        assert replay.operation_id == decision.operation_id and replay.cached_result == {"created": True}
    operations = await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")
    assert len(operations) == 1 and operations[0].agent_run_id == "resumed"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_bound_leaf_callbacks_keep_native_event_identity_and_permitted_payload(state: Persistence, failed: bool) -> None:
    from linktools.ai.capability import ToolCallFailed

    bridge = _bridge(state)
    repository = state.checkpoint.tools
    events = []
    new_flags = []

    async def admission(request, *, observed, event, known_new):
        new_flags.append(known_new)
        await repository.admit(request)
        events.append(event)
        return await repository.observe_operation(request.tool_operation_id, tenant_id="tenant")

    async def terminal(operation_id, *, tenant_id, owner, fence, observed, event,
                       result_payload=None, error_code=None, error_payload=None):
        assert observed is not None
        if result_payload is not None:
            await repository.complete_payload(operation_id, tenant_id=tenant_id, owner=owner,
                                              fence=fence, result_payload=result_payload)
        else:
            await repository.fail_payload(operation_id, tenant_id=tenant_id, owner=owner, fence=fence,
                                          error_code=error_code, error_payload=error_payload)
        events.append(event)
        return await repository.observe_operation(operation_id, tenant_id=tenant_id), False

    bridge.bind_boundaries(admission=admission, terminal=terminal)
    bridge.confirm_native_run(RunRecord(run_id="run", parent_run_id="parent", agent_name="worker",
                                       metadata={"native": "metadata"}))
    ctx = replace(tool_run_context(), run_step=7, conversation_id="conversation")
    call = ToolCallPart("inspect", {}, tool_call_id="call")
    decision = await bridge.begin(ctx, call, ToolDefinition(name="inspect"), {}, False)
    assert decision is not None
    effect = await bridge.get_tool_effect(run_id="run", tool_call_id="call")
    assert effect.status == "started"
    await bridge.record_tool_effect(replace(effect, effect_summary="Authorized provider call"))
    if failed:
        await bridge.fail(decision, ToolCallFailed("permitted failure"))
    else:
        await bridge.complete(decision, {"permitted": [False, None, 42]})
    assert new_flags == [True]
    assert [event.kind for event in events] == ["tool_call_started", "tool_call_failed" if failed else "tool_call_completed"]
    assert all(event.run_id == "run" and event.step_index == 7 and event.conversation_id == "conversation"
               and event.tool_name == "inspect" and event.tool_call_id == "call" for event in events)
    assert all(event.parent_run_id == "parent" and event.agent_name == "worker"
               and event.metadata == {"native": "metadata", "linktools.operation_id": decision.operation_id} for event in events)
    assert len({event.idempotency_key for event in events}) == 2
    assert (await bridge.get_tool_effect(run_id="run", tool_call_id="call")).effect_summary == "Authorized provider call"
    assert len(await repository.list_by_execution("execution", tenant_id="tenant")) == 1


@pytest.mark.asyncio
async def test_failed_admission_retry_does_not_reassert_known_new(state: Persistence) -> None:
    bridge = _bridge(state)
    flags = []

    async def admission(request, *, observed, event, known_new):
        flags.append(known_new)
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def terminal(*args, **kwargs):
        raise AssertionError("no admitted operation")

    bridge.bind_boundaries(admission=admission, terminal=terminal)
    for _ in range(2):
        with pytest.raises(AIError) as raised:
            await bridge.begin(tool_run_context(), ToolCallPart("inspect", {}, tool_call_id="call"),
                               ToolDefinition(name="inspect"), {}, False)
        assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert flags == [True, False]
    with pytest.raises(RuntimeError, match="admitted leaf"):
        await bridge.record_tool_effect(ToolEffectRecord(run_id="run", tool_call_id="call",
                                                        tool_name="inspect", status="started"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", (False, True))
async def test_split_terminal_event_failure_repairs_without_repeating_effect(
    state: Persistence, failed: bool,
) -> None:
    from linktools.ai.capability import ToolCallFailed

    class InterruptedEvents(InMemoryStepStore):
        interrupted = False

        async def append_event(self, event):
            if event.kind in {"tool_call_completed", "tool_call_failed"} and not self.interrupted:
                self.interrupted = True
                raise OSError("native event store unavailable after effect commit")
            await super().append_event(event)

    native = InterruptedEvents()
    bridge = _bridge(state)
    RuntimeStepStore(native, bridge, native_boundary=True)
    call = ToolCallPart("create", {}, tool_call_id="call")
    definition = ToolDefinition(name="create")
    decision = await bridge.begin(tool_run_context(), call, definition, {}, False)
    assert decision is not None and not decision.has_cached_result
    if failed:
        await bridge.fail(decision, ToolCallFailed("allowed failure"))
    else:
        await bridge.complete(decision, {"allowed": [False, None, 42]})
    expected_kind = "tool_call_failed" if failed else "tool_call_completed"
    assert [event.kind for event in await native.list_events(run_id="run")] == ["tool_call_started", expected_kind]
    hot_replay = await bridge.begin(replace(tool_run_context(), run_step=9), call, definition, {}, False)
    assert hot_replay is not None and hot_replay.operation_id == decision.operation_id
    assert [event.kind for event in await native.list_events(run_id="run")] == [
        "tool_call_started", expected_kind, "tool_call_started", expected_kind,
    ]
    resumed = _bridge(state, run_id="resumed", source_run_id="run")
    RuntimeStepStore(native, resumed, native_boundary=True)
    replay = await resumed.begin(tool_run_context(), call, definition, {}, False)
    assert replay is not None and replay.operation_id == decision.operation_id
    if failed:
        assert replay.cached_error.message == "allowed failure"
    else:
        assert replay.has_cached_result and replay.cached_result == {"allowed": [False, None, 42]}
    events = await native.list_events(run_id="resumed")
    assert [event.kind for event in events] == ["tool_call_started", expected_kind]
    assert events[-1].error == ("allowed failure" if failed else None)
    assert len(await state.checkpoint.tools.list_by_execution("execution", tenant_id="tenant")) == 1


@pytest.mark.asyncio
async def test_unknown_effect_repairs_native_failure_event_without_resolving_outcome(state: Persistence) -> None:
    class InterruptedEvents(InMemoryStepStore):
        interrupted = False

        async def append_event(self, event):
            if event.kind == "tool_call_failed" and not self.interrupted:
                self.interrupted = True
                raise OSError("native event store unavailable")
            await super().append_event(event)

    native = InterruptedEvents()
    bridge = _bridge(state)
    RuntimeStepStore(native, bridge, native_boundary=True)
    call, definition = ToolCallPart("create", {}, tool_call_id="call"), ToolDefinition(name="create")
    decision = await bridge.begin(tool_run_context(), call, definition, {}, False)
    with pytest.raises(AIError) as failure:
        await bridge.unknown(decision, RuntimeError("forbidden raw provider error"))
    assert failure.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    operations = await bridge.list_operations()
    assert len(operations) == 1 and operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
    events = await native.list_events(run_id="run")
    assert [event.kind for event in events] == ["tool_call_started", "tool_call_failed"]
    assert events[-1].error == ErrorCode.TOOL_EFFECT_UNKNOWN.value
    assert operations[0].effect_summary is None
    resumed = _bridge(state, run_id="resumed", source_run_id="run")
    RuntimeStepStore(native, resumed, native_boundary=True)
    with pytest.raises(AIError) as recovery:
        await resumed.begin(tool_run_context(), call, definition, {}, False)
    assert recovery.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert [event.kind for event in await native.list_events(run_id="resumed")] == ["tool_call_started", "tool_call_failed"]
    assert len(await resumed.list_operations()) == 1
