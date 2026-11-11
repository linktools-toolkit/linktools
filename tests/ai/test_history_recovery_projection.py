#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace observations and official recovery snapshots stay independent."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition

from linktools.ai.persistence import Persistence
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import HmacCursorSigner, agent_run_id, agent_conversation_id
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from tests.ai._runtime_test_helpers import native_checkpoint, seed_native_snapshot
from linktools.ai.history._projection import StepExecutionHistoryReader
from linktools.ai.observe.journal import ModelRequestJournal
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._model_interaction import StagedModelInteraction, build_inline_context_projection
from hashlib import sha256
from linktools.ai.persistence._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from linktools.ai.persistence.runs import RuntimeAgentRunStore, StagingAgentRunStore
from linktools.ai.storage import PayloadPolicy
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite",))
async def test_trace_reopen_keeps_both_responses_when_native_snapshot_is_older(
    tmp_path: Path, backend: str
) -> None:
    namespace = "public-ahead-proof"
    run_id = agent_run_id(
        namespace=namespace,
        tenant_id="tenant",
        execution_id="execution",
        agent_run_seq=1,
    )
    conversation = agent_conversation_id(
        namespace=namespace, tenant_id="tenant", execution_id="execution"
    )
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=namespace, tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        await repository.create_execution(execution)
        restored = state.run_store
        public = restored.read_store(PersistenceDomain.TRACE)
        recovery = restored.read_store(PersistenceDomain.CHECKPOINT)
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(), execution_repository=state.execution.executions, tenant_id="tenant",
            conversation_archive=restored.read_store(PersistenceDomain.CONVERSATION),
            execution_archive=public,
        )
        old.bind_step_store(native_checkpoint().store)
        await old.initialize()

        def recorder_for(store, generation, **kwargs):
            store.bind_execution_producer(
                run_id, execution_id="execution", producer_generation=generation, restart_checkpoints=False
            )

            async def boundary():
                await store.flush_execution_projection(
                    run_id, execution_id="execution", producer_generation=generation
                )

            async def checkpoint(value):
                await seed_native_snapshot(
                    store.step_store, run_id=value.agent_run_id,
                    messages=list(value.context_messages or value.messages),
                    step_index=value.step_index, state=value.state,
                    conversation_id=conversation, agent_name="agent", metadata={"agent_run_seq": "1"},
                )
                await store.save_checkpoint(
                    value, execution_id="execution", producer_generation=generation
                )

            return AgentRunRecorder(
                store,
                execution_id="execution",
                agent_run_id=run_id,
                history_boundary=boundary,
                checkpoint_sink=checkpoint,
                **kwargs,
            )

        run = AgentRunRecord(
            run_id,
            agent_id="agent",
            agent_conversation_id=conversation,
            metadata={"agent_run_seq": "1"},
        )
        original = recorder_for(old, 1)
        await original.register_agent_run(run)
        user = ModelRequest(run_id=run_id, parts=[UserPromptPart("original request")])
        a = ModelResponse(
            run_id=run_id, parts=[
                TextPart("response A before checkpoint"),
                ToolCallPart("tool", {"version": "a"}, tool_call_id="unused-a"),
            ]
        )
        original.append_transcript_message(user)
        await original.save_checkpoint(
            AgentRunCheckpoint(
                run_id,
                1,
                [user],
                context_messages=[user],
                transcript_message_count_before=0,
            )
        )
        model = TestModel()

        async def publish(recorder, journal, response, messages):
            fact = journal.begin(1)
            recorder.begin_model_interaction(
                fact, model, messages, None, ModelRequestParameters(), False
            )
            recorder.prepare_model_interaction(
                fact, model, messages, None, ModelRequestParameters(), False
            )
            await recorder.record_model_event(
                fact, phase="started", include_observation=False
            )
            fact = journal.finish(fact.model_request_seq, status="SUCCEEDED")
            recorder.finish_model_interaction(
                fact,
                model=model,
                response=response,
                status="SUCCEEDED",
                error_code=None,
                duration_ns=1,
                usage=None,
            )
            await recorder.record_model_event(
                fact, phase="completed", response=response, include_observation=False
            )
            return fact

        journal = ModelRequestJournal(
            source_namespace=namespace,
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id=run_id,
        )
        await publish(original, journal, a, [user])
        # Controlled crash point: response publication succeeded; before-tool recovery
        # checkpoint and tool dispatch have not happened. Discard old staging owner.
        assert (await recovery.latest_checkpoint(agent_run_id=run_id)).messages == [
            user
        ]
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [user, a]
        assert (
            await state.checkpoint.tools.list_by_agent_run(run_id, tenant_id="tenant")
            == ()
        )
        await repository.compare_and_swap(
            "execution",
            tenant_id="tenant",
            expected_revision=1,
            next_record=replace(execution, revision=2, producer_generation=2, producer_claim_id="replacement-claim"),
        )
        loaded = await restored.load_loaded_model_context(
            PersistenceDomain.CHECKPOINT, run_id
        )
        sdk_input = list(loaded.model_messages())
        assert sdk_input == [user]
        recorder = recorder_for(
            restored, 2, initial_messages=sdk_input, initial_context=loaded
        )
        ctx = SimpleNamespace(
            messages=sdk_input, conversation_id=conversation, run_step=2
        )
        await recorder.register_agent_run(run)
        assert recorder.transcript_messages() == (user, a)
        high_water = await restored.model_interaction_count(agent_run_id=run_id)
        assert high_water == 1
        resumed_journal = ModelRequestJournal(
            source_namespace=namespace,
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id=run_id,
            next_model_request_seq=high_water + 1,
        )
        recorder.append_transcript_message(ctx.messages[-1])
        await seed_native_snapshot(
            restored.step_store, run_id=run_id, messages=list(ctx.messages), step_index=ctx.run_step,
        )
        received = []
        b = ModelResponse(
            run_id=run_id, parts=[
                TextPart("different response B"),
                ToolCallPart("tool", {"version": "b"}, tool_call_id="effect-b"),
            ]
        )

        async def provider(messages, info):
            received.append(list(messages))
            assert messages == [user]
            return b

        response = await FunctionModel(provider).request(
            ctx.messages,
            None,
            ModelRequestParameters(
                function_tools=[ToolDefinition(name="tool")], allow_text_output=True
            ),
        )
        fact = await publish(recorder, resumed_journal, response, ctx.messages)
        assert fact.model_request_seq == 2
        assert received == [[user]]
        ctx.messages.append(response)
        await recorder.save_checkpoint(
            AgentRunCheckpoint(
                run_id,
                2,
                list(recorder.transcript_messages()),
                context_messages=ctx.messages,
                transcript_message_count_before=3,
            )
        )
        recovered = await recovery.latest_checkpoint(agent_run_id=run_id)
        assert recovered.messages == [user, response]
        assert recovered.context_messages == [user, response]
        commands = StateCommands(
            repository,
            namespace=namespace,
            events=state.execution.events,
            tools=state.checkpoint.tools,
            background_tasks=set(),
        )
        bridge = RuntimeToolOperationBridge(
            state.checkpoint.tools,
            state.object_store(PersistenceDomain.CHECKPOINT),
            namespace=namespace,
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id=run_id,
            binding_digest=execution.binding_digest,
            owner="replacement-claim",
            background_tasks=set(),
            payload_policy=PayloadPolicy(),
            terminal_commands=commands,
            producer_generation=2,
        )
        call = response.parts[1]
        effects = []
        for _ in range(2):
            bridge = RuntimeToolOperationBridge(
                state.checkpoint.tools,
                state.object_store(PersistenceDomain.CHECKPOINT),
                namespace=namespace,
                tenant_id="tenant",
                execution_id="execution",
                agent_run_id=run_id,
                binding_digest=execution.binding_digest,
                owner="replacement-claim",
                background_tasks=set(),
                payload_policy=PayloadPolicy(),
                terminal_commands=commands,
                producer_generation=2,
            )
            decision = await bridge.begin(
                SimpleNamespace(),
                call,
                ToolDefinition(name="tool"),
                call.args_as_dict(),
                True,
            )
            if decision.has_cached_result:
                result = decision.cached_result
            else:
                effects.append(call.tool_call_id)
                result = "B effect result"
                await bridge.complete(decision, result)
            await recorder.record_tool_result_boundary(
                ToolReturnPart("tool", result, tool_call_id=call.tool_call_id), 2
            )
        assert effects == ["effect-b"]
        reader = StepExecutionHistoryReader(
            namespace=namespace,
            executions=repository,
            store=public,
            cursor_signer=HmacCursorSigner(namespace, b"prefix-proof"),
        )
        page = await reader.history(
            "execution", tenant_id="tenant", limit=100, cursor=None
        )
        texts = [
            (x.content, x.message_seq, x.model_request_seq)
            for x in page.items
            if x.item_kind == "assistant"
        ]
        assert texts == [
            ("response A before checkpoint", 2, 1),
            ("different response B", 4, 2),
        ]
        calls = [
            (x.tool_call_id, x.message_seq, x.model_request_seq)
            for x in page.items
            if x.item_kind == "tool_call"
        ]
        assert calls == [("unused-a", 2, 1), ("effect-b", 4, 2)]
        assert [
            (x.tool_call_id, x.content)
            for x in page.items
            if x.item_kind == "tool_result"
        ] == [("effect-b", "B effect result")]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite",))
@pytest.mark.parametrize("published_state", (None, "RUNNING", "INTERRUPTED"))
async def test_trace_publication_is_atomic_and_does_not_invent_snapshot_associations(
    tmp_path: Path,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    published_state: str | None,
) -> None:
    namespace = "public-behind-proof"
    run_id = agent_run_id(
        namespace=namespace,
        tenant_id="tenant",
        execution_id="execution",
        agent_run_seq=1,
    )
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=namespace, tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        await repository.create_execution(execution)
        root_store = state.run_store
        store = RuntimeAgentRunStore(
            StagingAgentRunStore(), execution_repository=state.execution.executions, tenant_id="tenant",
            conversation_archive=root_store.read_store(PersistenceDomain.CONVERSATION),
            execution_archive=root_store.read_store(PersistenceDomain.TRACE),
        )
        store.bind_step_store(native_checkpoint().store)
        await store.initialize()
        public = store.read_store(PersistenceDomain.TRACE)
        recovery = store.read_store(PersistenceDomain.CHECKPOINT)
        run = AgentRunRecord(
            run_id,
            agent_id="agent",
            metadata={"agent_run_seq": "1"},
            agent_conversation_id=agent_conversation_id(
                namespace=namespace, tenant_id="tenant", execution_id="execution"
            ),
        )
        store.bind_execution_producer(
            run_id, execution_id="execution", producer_generation=1
        )
        await store.register_agent_run(run, execution_id="execution")
        user = ModelRequest(run_id=run_id, parts=[UserPromptPart("recoverable input")])
        await store.save_checkpoint(
            AgentRunCheckpoint(
                run_id,
                1,
                [user],
                context_messages=[user],
                transcript_message_count_before=0,
            ),
            execution_id="execution",
            producer_generation=1,
        )
        await seed_native_snapshot(
            store.step_store, run_id=run_id, messages=[user],
            conversation_id=run.agent_conversation_id, agent_name=run.agent_id, metadata=run.metadata,
        )
        await store.flush_execution_projection(
            run_id, execution_id="execution", producer_generation=1
        )
        response = ModelResponse(run_id=run_id, parts=[TextPart("completed while public lagged")])
        await seed_native_snapshot(
            store.step_store, run_id=run_id, messages=[user, response],
            conversation_id=run.agent_conversation_id, agent_name=run.agent_id, metadata=run.metadata,
        )
        snapshot = await store.step_store.latest_snapshot(run_id=run_id)
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [user]
        assert await public.model_interaction_count(agent_run_id=run_id) == 0
        assert await recovery.model_interaction_count(agent_run_id=run_id) == 0
        published_request = None
        if published_state is not None:
            payloads = {}

            def intern(raw):
                digest = sha256(raw).hexdigest()
                payloads[digest] = raw
                return digest, len(raw)

            unfinished = StagedModelInteraction(
                agent_run_id=run_id, step_index=1, model_request_seq=1, purpose="agent",
                output_retry_index=None, model={"model_name": "test"},
                request_context=build_inline_context_projection((user,), intern),
                request_envelope_digest=intern(b"{}")[0], response_context=None,
                status=published_state, error_code=None, duration_ns=None, usage=None,
                started_at=run.started_at, finished_at=None,
            )
            prepared = PreparedArchive()
            relocated = await public.prepare_interactions(
                run, (unfinished,), payloads.__getitem__, local_message_count=1,
                local_messages=(user,), prepared=prepared,
            )
            published_request = relocated[0].request_context
            await public.sync_projection(
                run, events=(), checkpoints=(), interactions=relocated,
                execution_id="execution", archive=prepared,
            )
        observed = AgentRunCheckpoint(run_id, 2, [user, response], transcript_message_count_before=1)
        root_store.bind_execution_producer(
            run_id, execution_id="execution", producer_generation=1
        )
        before_events = await public.list_events(agent_run_id=run_id)
        group = public.state_store.storage_group
        original_mutate = group.mutate
        fail = [True]

        async def failed_mutate(stores, callback):
            async def body(transaction):
                result = await callback(transaction)
                if fail[0]:
                    raise AIError(
                        ErrorCode.STORAGE_CONFLICT, "controlled pre-commit failure"
                    )
                return result

            return await original_mutate(stores, body)

        monkeypatch.setattr(group, "mutate", failed_mutate)
        with pytest.raises(AIError, match="controlled pre-commit failure"):
            await public.sync_projection(
                run, events=(), checkpoints=(observed,), execution_id="execution",
            )
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [user]
        assert await public.model_interaction_count(agent_run_id=run_id) == int(
            published_state is not None
        )
        if published_state is not None:
            assert (await public.list_model_interactions(agent_run_id=run_id))[
                0
            ].status == published_state
        fail[0] = False
        await public.sync_projection(
            run, events=(), checkpoints=(observed,), execution_id="execution",
        )
        assert await store.step_store.latest_snapshot(run_id=run_id) == snapshot
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [
            user,
            response,
        ]
        interactions = await public.list_model_interactions(agent_run_id=run_id)
        assert len(interactions) == int(published_state is not None)
        if published_state is not None:
            assert interactions[0].model_request_seq == 1
            assert interactions[0].status == published_state
            assert interactions[0].request_context == published_request
            assert interactions[0].response_context is None
        assert await public.list_events(agent_run_id=run_id) == before_events
        associations = await public.read_history_associations(
            agent_run_id=run_id,
            message_seqs=(2,),
            tool_call_ids=(),
            event_high_water=len(before_events),
        )
        assert not associations
        reader = StepExecutionHistoryReader(
            namespace=namespace,
            executions=repository,
            store=public,
            cursor_signer=HmacCursorSigner(namespace, b"behind-proof"),
        )
        page = await reader.history(
            "execution", tenant_id="tenant", limit=100, cursor=None
        )
        assistant = [x for x in page.items if x.item_kind == "assistant"]
        assert (
            len(assistant) == 1
            and assistant[0].content == "completed while public lagged"
        )
        assert assistant[0].message_seq == 2 and assistant[0].model_request_seq is None
        root_store.bind_execution_producer(
            run_id, execution_id="execution", producer_generation=1
        )
        await root_store.register_agent_run(run, execution_id="execution")
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite",))
async def test_observed_completed_tool_group_consumes_published_pending_results(
    tmp_path: Path,
    backend: str,
) -> None:
    from linktools.ai.persistence._transcript_staging import StagedTranscript

    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="restored-tool-group", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        await repository.create_execution(execution)
        restored = state.run_store
        public = restored.read_store(PersistenceDomain.TRACE)
        recovery = restored.read_store(PersistenceDomain.CHECKPOINT)
        run = AgentRunRecord("run")
        user = ModelRequest(run_id="run", parts=[UserPromptPart("run tools")])
        response = ModelResponse(
            run_id="run", parts=[
                ToolCallPart("tool", {}, tool_call_id="quick"),
                ToolCallPart("tool", {}, tool_call_id="slow"),
            ]
        )
        quick = ToolReturnPart("tool", "quick result", tool_call_id="quick")
        slow = ToolReturnPart("tool", "slow result", tool_call_id="slow")
        completed = ModelRequest(run_id="run", parts=[quick, slow])
        before = AgentRunCheckpoint(
            "run", 1, [user, response], transcript_message_count_before=0
        )
        await public.sync_projection(
            run, events=(), checkpoints=(before,), execution_id="execution"
        )
        await seed_native_snapshot(
            restored.step_store, run_id=run.agent_run_id, messages=before.messages,
        )
        pending = await public.transcript_repository.prepare_observation(
            "run",
            (),
            first_message_index=2,
            pending=ModelRequest(run_id="run", parts=[quick]),
            pending_keys=("tool_result:quick",),
        )
        await public.sync_prepared_projection(
            run,
            events=(),
            checkpoints=(),
            observation=pending,
            execution_id="execution",
            producer_generation=1,
        )
        await seed_native_snapshot(
            restored.step_store, run_id=run.agent_run_id,
            messages=[user, response, completed], step_index=2,
        )
        observed = await public.transcript_repository.prepare_observation(
            "run", (completed,), first_message_index=2, pending=None, pending_keys=(),
        )
        await public.sync_prepared_projection(
            run, events=(), checkpoints=(), observation=observed,
            execution_id="execution", producer_generation=1,
        )
        restored.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1, restart_checkpoints=False,
        )
        await restored.register_agent_run(run, execution_id="execution")
        assert restored.staged_transcript("run") == StagedTranscript(
            (user, response, completed)
        )
        head = await public.transcript_repository.get_head("run")
        assert (
            head.message_count == 3
            and head.pending is None
            and head.pending_part_count == 0
        )
        assert [m async for m in public.iter_messages(agent_run_id="run")] == [
            user,
            response,
            completed,
        ]
        capture = (await public.capture_history(("run",), include_pending=True))["run"]
        assert capture.message_count == 3
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite",))
async def test_trace_checkpoint_rejects_replaced_producer_before_public_write(
    tmp_path: Path,
    backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="stale-recovery-checkpoint", tenant_id="tenant")
    state.run_store.bind_step_store(native_checkpoint().store)
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim")
        await repository.create_execution(execution)
        root = state.run_store
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(), execution_repository=state.execution.executions, tenant_id="tenant",
            conversation_archive=root.read_store(PersistenceDomain.CONVERSATION),
            execution_archive=root.read_store(PersistenceDomain.TRACE),
        )
        old.bind_step_store(native_checkpoint().store)
        await old.initialize()
        old.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1
        )
        await old.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        await repository.compare_and_swap(
            "execution",
            tenant_id="tenant",
            expected_revision=1,
            next_record=replace(execution, revision=2, producer_generation=2, producer_claim_id="replacement-claim"),
        )
        await old.save_checkpoint(
            AgentRunCheckpoint(
                "run",
                1,
                [ModelRequest(parts=[UserPromptPart("stale")])],
                transcript_message_count_before=0,
            ),
            execution_id="execution",
            producer_generation=1,
        )
        with pytest.raises(AIError) as stale:
            await old.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert (
            await root.read_store(PersistenceDomain.CHECKPOINT).latest_checkpoint(
                agent_run_id="run"
            )
            is None
        )
        assert (
            await root.read_store(PersistenceDomain.TRACE).latest_checkpoint(
                agent_run_id="run"
            )
            is None
        )
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("publication", ("missing", "pending", "complete"))
@pytest.mark.parametrize("restamped", (False, True))
async def test_recovered_request_is_prior_attempt_input_not_new_raw_occurrence(publication: str, restamped: bool) -> None:
    from linktools.ai.persistence._transcript_staging import StagedTranscript

    store = StagingAgentRunStore()
    await store.initialize()
    request = ModelRequest(parts=[ToolReturnPart("tool", "same value", tool_call_id="old-call")], run_id="old")
    await store.register_agent_run(AgentRunRecord("old"))
    if publication == "pending":
        store.stage_transcript("old", StagedTranscript((), request, ("tool_result:old-call",)))
    elif publication == "complete":
        store.stage_transcript("old", StagedTranscript((request,)))
    before = store.staged_transcript("old")
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="new", initial_messages=(request,))
    await recorder.register_agent_run(AgentRunRecord("new"))
    if restamped:
        request = replace(request, timestamp=datetime(2001, 1, 1, tzinfo=timezone.utc))
    assert recorder.request_already_captured((request,))
    response = ModelResponse(parts=[TextPart("resumed response")], run_id="new")
    recorder.append_transcript_message(response)
    assert recorder.transcript_messages() == (response,)
    assert store.staged_transcript("old") == before


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", (None, "new", "old"))
async def test_new_tool_return_occurrences_are_not_suppressed_by_equal_contents(origin: str | None) -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    original = ModelRequest(parts=[ToolReturnPart("tool", "same value", tool_call_id="old-call")], run_id=origin)
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="new", initial_messages=(original,))
    await recorder.register_agent_run(AgentRunRecord("new"))
    request = (replace(original, parts=[ToolReturnPart("tool", "same value", tool_call_id="new-call")], run_id="new")
               if origin == "old" else original)
    assert not recorder.request_already_captured((request,))
    recorder.append_transcript_message(request)
    assert recorder.transcript_messages() == (request,)


@pytest.mark.asyncio
async def test_initial_request_occurrence_is_frozen_and_consumed_only_once() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    original = ModelRequest(parts=[ToolReturnPart("tool", "value", tool_call_id="call")], run_id="old")
    initial = replace(original, parts=list(original.parts))
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="new", initial_messages=(original,))
    original.parts.append(UserPromptPart("later SDK mutation"))
    await recorder.register_agent_run(AgentRunRecord("new"))
    assert recorder.request_already_captured((initial,))
    assert not recorder.request_already_captured((initial,))
    later = replace(initial, run_id="new")
    recorder.append_transcript_message(later)
    assert recorder.transcript_messages() == (later,)


@pytest.mark.asyncio
async def test_recovered_request_with_new_run_identity_remains_a_new_occurrence() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    initial = ModelRequest(parts=[ToolReturnPart("tool", "same", tool_call_id="call")], run_id="old")
    recorder = AgentRunRecorder(store, execution_id="execution", agent_run_id="new", initial_messages=(initial,))
    await recorder.register_agent_run(AgentRunRecord("new"))
    assert not recorder.request_already_captured((replace(initial, run_id="new"),))
