#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observed tool-result prefixes survive replay from an older checkpoint."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._capabilities import _AgentRunPersistenceCapability
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime.state._plan import RuntimeRetentionMode
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.runtime.state._steps import RuntimeAgentRunStore, StagingAgentRunStore

from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_partial", (False, True))
@pytest.mark.parametrize("conflicting_replay", (False, True))
async def test_recovery_preserves_each_tool_result_once_with_its_original_position(
    completed_partial: bool, conflicting_replay: bool,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="partial-recovery", tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        first = state.run_store
        first.bind_execution_producer("run", execution_id="execution", producer_generation=1)

        def recorder_for(store):
            async def boundary():
                await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)

            async def checkpoint_sink(value):
                await store.save_checkpoint(value, execution_id="execution", producer_generation=1)

            return AgentRunRecorder(
                store, execution_id="execution", agent_run_id="run",
                history_boundary=boundary, checkpoint_sink=checkpoint_sink,
            )

        original = recorder_for(first)
        await original.register_agent_run(AgentRunRecord("run", agent_id="agent", agent_conversation_id="conversation"))
        user = ModelRequest(parts=[UserPromptPart("run both")])
        response = ModelResponse(parts=[
            ToolCallPart("tool", {}, tool_call_id="quick"),
            ToolCallPart("tool", {}, tool_call_id="slow"),
        ])
        original.append_transcript_message(user)
        journal = ModelRequestJournal(
            source_namespace="partial-recovery", tenant_id="tenant",
            execution_id="execution", agent_run_id="run",
        )
        model = TestModel()
        fact = journal.begin(1)
        original.begin_model_interaction(fact, model, [user], None, ModelRequestParameters(), False)
        original.prepare_model_interaction(fact, model, [user], None, ModelRequestParameters(), False)
        await original.record_model_event(fact, phase="started", include_observation=False)
        fact = journal.finish(fact.model_request_seq, status="SUCCEEDED")
        original.finish_model_interaction(
            fact, model=model, response=response, status="SUCCEEDED", error_code=None,
            duration_ns=1, usage=None,
        )
        await original.record_model_event(fact, phase="completed", response=response, include_observation=False)
        await original.save_checkpoint(AgentRunCheckpoint(
            "run", 1, [user, response], transcript_message_count_before=0,
        ))
        quick = ToolReturnPart("tool", "quick done", tool_call_id="quick")
        slow = ToolReturnPart("tool", "slow done", tool_call_id="slow")
        await original.record_tool_result_boundary(quick, 1)
        if completed_partial:
            original.finish_transcript(interrupted=True)
            await original.commit_history_boundary()
        archive = first.read_store(RuntimeDomain.EXECUTION)
        before = (await archive.capture_history(("run",), include_pending=True))["run"]
        assert before.message_count == 3

        second = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=first.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=archive,
            recovery_archive=first.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await second.initialize()
        second.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        recorder = recorder_for(second)
        capability = _AgentRunPersistenceCapability(recorder=recorder, agent_id="agent", agent_run_id="run")
        ctx = SimpleNamespace(messages=[user, response], conversation_id="conversation", run_step=1)
        await capability.before_run(ctx)
        replayed = replace(
            quick, timestamp=quick.timestamp + timedelta(seconds=1),
            content="changed output" if conflicting_replay else quick.content,
        )
        if conflicting_replay:
            with pytest.raises(AIError) as caught:
                await recorder.record_tool_result_boundary(replayed, 1)
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            return
        await recorder.record_tool_result_boundary(replayed, 1)
        await recorder.record_tool_result_boundary(slow, 1)
        ctx.messages.append(ModelRequest(parts=[replayed, slow]))
        await capability.before_model_request(ctx, SimpleNamespace())
        recorder.append_transcript_message(ModelResponse(parts=[TextPart("done")]))
        await recorder.commit_history_boundary()
        messages = [message async for message in archive.iter_messages(agent_run_id="run")]
        results = [(message_index, part_index, part) for message_index, message in enumerate(messages)
                   for part_index, part in enumerate(message.parts) if isinstance(part, ToolReturnPart)]
        assert [(part.tool_call_id, part.content) for _, _, part in results] == [("quick", "quick done"), ("slow", "slow done")]
        assert results[0] == (2, 0, quick)
        assert results[1][:2] == ((3, 0) if completed_partial else (2, 1))
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", (False, True))
@pytest.mark.parametrize("conflicting_call", (False, True))
async def test_replayed_model_calls_reuse_effects_and_keep_new_raw_part_indexes(
    streamed: bool, conflicting_call: bool,
) -> None:
    from pydantic_ai.tools import ToolDefinition

    from linktools.ai.core import HmacCursorSigner, agent_conversation_id as make_conversation_id, agent_run_id as make_agent_run_id
    from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
    from linktools.ai.runtime._tool import RuntimeToolOperationBridge
    from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
    from linktools.ai.storage import PayloadPolicy

    namespace = "replayed-model"
    run_id = make_agent_run_id(
        namespace=namespace, tenant_id="tenant", execution_id="execution", agent_run_seq=1,
    )
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        commands = RuntimeStateCommands(
            repository, namespace=namespace, events=state.execution.events,
            tools=state.recovery.tools, background_tasks=set(),
        )

        def bridge_for(owner):
            return RuntimeToolOperationBridge(
                state.recovery.tools, state.object_store(RuntimeDomain.RECOVERY),
                namespace=namespace, tenant_id="tenant", execution_id="execution",
                agent_run_id=run_id, binding_digest=execution.binding_digest,
                owner=owner, background_tasks=set(), payload_policy=PayloadPolicy(),
                terminal_commands=commands, producer_generation=1,
            )

        def recorder_for(store):
            store.bind_execution_producer(run_id, execution_id="execution", producer_generation=1)

            async def boundary():
                await store.flush_execution_projection(run_id, execution_id="execution", producer_generation=1)

            async def checkpoint_sink(value):
                await store.save_checkpoint(value, execution_id="execution", producer_generation=1)

            return AgentRunRecorder(
                store, execution_id="execution", agent_run_id=run_id,
                history_boundary=boundary, checkpoint_sink=checkpoint_sink,
            )

        user = ModelRequest(parts=[UserPromptPart("run tools")])
        quick_call = ToolCallPart("tool", {"which": "quick"}, tool_call_id="quick")
        slow_call = ToolCallPart("tool", {"which": "slow"}, tool_call_id="slow")
        new_call = ToolCallPart("tool", {"which": "new"}, tool_call_id="new")
        first_response = ModelResponse(parts=[quick_call, slow_call])
        original = recorder_for(state.run_store)
        run = AgentRunRecord(
            run_id, agent_id="agent", metadata={"agent_run_seq": "1"},
            agent_conversation_id=make_conversation_id(
                namespace=namespace, tenant_id="tenant", execution_id="execution",
            ),
        )
        await original.register_agent_run(run)
        original.append_transcript_message(user)
        await original.save_checkpoint(AgentRunCheckpoint(run_id, 1, [user], transcript_message_count_before=0))
        journal = ModelRequestJournal(
            source_namespace=namespace, tenant_id="tenant", execution_id="execution", agent_run_id=run_id,
        )
        model = TestModel()

        async def publish_response(recorder, response, *, stream=False):
            fact = journal.begin(1)
            recorder.begin_model_interaction(fact, model, [user], None, ModelRequestParameters(), stream)
            recorder.prepare_model_interaction(fact, model, [user], None, ModelRequestParameters(), stream)
            await recorder.record_model_event(fact, phase="started", include_observation=False)
            if stream:
                for index, part in enumerate(response.parts):
                    recorder.stage_response_part(part, index)
                    await recorder.commit_history_boundary()
            fact = journal.finish(fact.model_request_seq, status="SUCCEEDED")
            recorder.finish_model_interaction(
                fact, model=model, response=response, status="SUCCEEDED", error_code=None,
                duration_ns=1, usage=None,
            )
            await recorder.record_model_event(fact, phase="completed", response=response, include_observation=False)

        await publish_response(original, first_response)
        effects = []

        async def execute(bridge, recorder, call):
            decision = await bridge.begin(SimpleNamespace(), call, ToolDefinition(name="tool"), call.args_as_dict(), True)
            if decision.has_cached_result:
                result = decision.cached_result
            else:
                effects.append(call.tool_call_id)
                result = f"{call.tool_call_id} result"
                await bridge.complete(decision, result)
            part = ToolReturnPart("tool", result, tool_call_id=call.tool_call_id)
            await recorder.record_tool_result_boundary(part, 1)
            return part

        quick_result = await execute(bridge_for("first"), original, quick_call)
        second = RuntimeAgentRunStore(
            StagingAgentRunStore(), conversation_archive=state.run_store.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=archive, recovery_archive=state.run_store.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await second.initialize()
        recorder = recorder_for(second)
        await recorder.register_agent_run(run)
        replayed_response = ModelResponse(parts=[
            replace(quick_call, args={"which": "changed"}) if conflicting_call else quick_call,
            TextPart("new text"), slow_call, new_call,
        ])
        if conflicting_call:
            with pytest.raises(AIError) as caught:
                await publish_response(recorder, replayed_response, stream=streamed)
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            assert effects == ["quick"]
            return
        await publish_response(recorder, replayed_response, stream=streamed)
        resumed = bridge_for("resumed")
        results = [await execute(resumed, recorder, call) for call in (quick_call, slow_call, new_call)]
        recorder.append_transcript_message(ModelRequest(parts=results))
        await recorder.commit_history_boundary()
        assert effects == ["quick", "slow", "new"]
        messages = [message async for message in archive.iter_messages(agent_run_id=run_id)]
        assert messages[2].parts == [quick_result]
        assert messages[3] == replayed_response
        assert [(part.tool_call_id, part.content) for message in messages for part in message.parts
                if isinstance(part, ToolReturnPart)] == [("quick", "quick result"), ("slow", "slow result"), ("new", "new result")]
        reader = StepExecutionHistoryReader(
            namespace=namespace, executions=repository, store=archive,
            cursor_signer=HmacCursorSigner(namespace, b"replay-key"),
        )
        history = await reader.history("execution", tenant_id="tenant", limit=200, cursor=None)
        calls = [item for item in history.items if item.item_kind == "tool_call"]
        assert [(item.tool_call_id, item.message_seq, item.part_index, item.model_request_seq) for item in calls] == [
            ("quick", 2, 0, 1), ("slow", 2, 1, 1), ("new", 4, 3, 2),
        ]
        assert [(item.message_seq, item.part_index) for item in history.items if item.content == "new text"] == [(4, 1)]
        third = recorder_for(second)
        await third.register_agent_run(run)
        assert third.model_request_seq_for_tool_call("quick") == 1
        assert third.model_request_seq_for_tool_call("new") == 2
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_delayed_observation_cannot_write_after_another_producer_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="delayed-producer", tenant_id="tenant")
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        first = state.run_store
        first.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        run = AgentRunRecord("run", agent_id="agent", agent_conversation_id="conversation")
        await first.register_agent_run(run, execution_id="execution")
        await first.save_checkpoint(
            AgentRunCheckpoint("run", 1, [ModelRequest(parts=[UserPromptPart("start")])], transcript_message_count_before=0),
            execution_id="execution", producer_generation=1,
        )
        await first.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        archive = first.read_store(RuntimeDomain.EXECUTION)
        old_producer = RuntimeAgentRunStore(
            StagingAgentRunStore(), conversation_archive=first.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=archive, recovery_archive=first.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await old_producer.initialize()
        old_producer.bind_execution_producer("run", execution_id="execution", producer_generation=1)

        async def delayed_boundary():
            await old_producer.flush_execution_projection(
                "run", execution_id="execution", producer_generation=1, deferred=True,
            )

        recorder = AgentRunRecorder(
            old_producer, execution_id="execution", agent_run_id="run", history_boundary=delayed_boundary,
        )
        await recorder.register_agent_run(run)
        completed_attempt = asyncio.Event()
        original_commit = old_producer.commit_captured_execution_projection

        async def observe_attempt(*args, **kwargs):
            try:
                return await original_commit(*args, **kwargs)
            finally:
                completed_attempt.set()

        monkeypatch.setattr(old_producer, "commit_captured_execution_projection", observe_attempt)
        await recorder.record_tool_start(ToolCallPart("tool", {}, tool_call_id="late"), 1)
        assert await archive.list_events(agent_run_id="run") == []
        updated = await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=1,
            next_record=replace(execution, revision=2),
        )
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, updated)
        )
        await asyncio.wait_for(completed_attempt.wait(), 5)
        assert await archive.list_events(agent_run_id="run") == []
        with pytest.raises(AIError) as caught:
            await old_producer.flush_execution_projection(
                "run", execution_id="execution", producer_generation=1,
            )
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        old_producer.bind_execution_producer("run", execution_id="execution", producer_generation=2)
        with pytest.raises(AIError) as rebound:
            await old_producer.flush_execution_projection(
                "run", execution_id="execution", producer_generation=2,
            )
        assert rebound.value.code is ErrorCode.STORAGE_CONFLICT
        with pytest.raises(AIError) as discard:
            await old_producer.discard_revoked_producer_staging(
                "run", execution_id="execution", producer_generation=1,
            )
        assert discard.value.code is ErrorCode.STORAGE_CONFLICT
        assert old_producer._projection_dirty == {"run"}
        assert await old_producer.get_agent_run(agent_run_id="run") == run
        assert await archive.list_events(agent_run_id="run") == []
        with pytest.raises(AIError) as close_error:
            await old_producer.preflight_close()
        assert close_error.value.code is ErrorCode.STORAGE_CONFLICT
        assert (await repository.get_history_head("execution", tenant_id="tenant")).producer_generation == 2
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ("cancel_checkpoint", "failed_observation"))
async def test_recovery_checkpoint_settles_and_public_drain_retries_interrupted_observation(
    monkeypatch: pytest.MonkeyPatch, interrupt: str,
) -> None:
    import asyncio

    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="observation-checkpoint", tenant_id="tenant")
    release = asyncio.Event()
    checkpoint_task = None
    try:
        execution = replace(_execution(), revision=1)
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        store = state.run_store
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
        await store.register_agent_run(AgentRunRecord("run", agent_id="agent"), execution_id="execution")
        first = ModelRequest(parts=[UserPromptPart("start")])
        checkpoint = AgentRunCheckpoint("run", 1, [first], transcript_message_count_before=0)
        await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        archive = store.read_store(RuntimeDomain.EXECUTION)
        entered = asyncio.Event()
        original = archive.sync_prepared_projection
        intercepted = False

        async def interrupt_observation(*args, **kwargs):
            nonlocal intercepted
            if not intercepted:
                intercepted = True
                entered.set()
                if interrupt == "failed_observation":
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
                await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(archive, "sync_prepared_projection", interrupt_observation)
        recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="run")
        await recorder.register_agent_run(AgentRunRecord("run", agent_id="agent"))
        await recorder.record_event("AGENT_RUN_STARTED", 1)
        await store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1, deferred=True,
        )
        await asyncio.wait_for(entered.wait(), 5)
        target = replace(checkpoint, step_index=2, state="interrupted", transcript_message_count_before=1)
        if interrupt == "failed_observation":
            await store.wait_projection_flight("run")
            await asyncio.sleep(0)
            with pytest.raises(AIError) as failed:
                await store.flush_execution_projection(
                    "run", execution_id="execution", producer_generation=1, deferred=True,
                )
            assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
        checkpoint_task = asyncio.create_task(store.save_checkpoint(
            target, execution_id="execution", producer_generation=1,
        ))
        if interrupt == "cancel_checkpoint":
            await asyncio.sleep(0)
            checkpoint_task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await checkpoint_task
        else:
            await checkpoint_task
        recovered = await store.read_store(RuntimeDomain.RECOVERY).latest_checkpoint(
            agent_run_id="run", include_interrupted=True,
        )
        assert recovered is not None and recovered.step_index == 2 and recovered.state == "interrupted"
        if interrupt == "failed_observation":
            assert "run" in store._projection_dirty
            assert await archive.list_events(agent_run_id="run") == []
        await store.flush_execution_projection(
            "run", execution_id="execution", producer_generation=1,
        )
        assert [event.event_type for event in await archive.list_events(agent_run_id="run")] == ["AGENT_RUN_STARTED"]
    finally:
        release.set()
        if checkpoint_task is not None:
            await asyncio.gather(checkpoint_task, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
async def test_terminal_producer_discard_preserves_unresolved_durability_flight() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="unresolved-producer", tenant_id="tenant")
    store = state.run_store
    store.bind_execution_producer("run", execution_id="execution", producer_generation=1)
    run = AgentRunRecord("run")
    await store.register_agent_run(run, execution_id="execution")
    captured = await store.capture_execution_projection("run")
    assert captured is not None
    _, flight = captured
    unknown = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    await store._fence_durability_flight(flight, unknown)
    with pytest.raises(AIError) as caught:
        await store.discard_revoked_producer_staging(
            "run", execution_id="execution", producer_generation=1,
        )
    assert caught.value is unknown
    assert store._durability_flights["run"] is flight
    assert store._projection_dirty == {"run"}
    assert await store.get_agent_run(agent_run_id="run") == run
    with pytest.raises(AIError) as close_error:
        await state.close()
    assert close_error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
