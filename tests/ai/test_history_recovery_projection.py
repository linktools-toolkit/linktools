#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled public-ahead restart proof without production separation changes."""

from dataclasses import replace
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

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.core import HmacCursorSigner, agent_run_id, agent_conversation_id
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._capabilities import _AgentRunPersistenceCapability
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.runtime.state._plan import RuntimeRetentionMode
from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
from linktools.ai.runtime.state._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from linktools.ai.runtime.state._steps import RuntimeAgentRunStore, StagingAgentRunStore
from linktools.ai.storage import PayloadPolicy
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_public_response_ahead_then_distinct_sdk_response(
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
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        restored = state.run_store
        public = restored.read_store(RuntimeDomain.EXECUTION)
        recovery = restored.read_store(RuntimeDomain.RECOVERY)
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=restored.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=public,
            recovery_archive=recovery,
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await old.initialize()

        def recorder_for(store, generation, **kwargs):
            store.bind_execution_producer(
                run_id, execution_id="execution", producer_generation=generation
            )

            async def boundary():
                await store.flush_execution_projection(
                    run_id, execution_id="execution", producer_generation=generation
                )

            async def checkpoint(value):
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
        user = ModelRequest(parts=[UserPromptPart("original request")])
        a = ModelResponse(
            parts=[
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
            await state.recovery.tools.list_by_agent_run(run_id, tenant_id="tenant")
            == ()
        )
        updated = await repository.compare_and_swap(
            "execution",
            tenant_id="tenant",
            expected_revision=1,
            next_record=replace(execution, revision=2),
        )
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, updated)
        )
        loaded = await restored.load_loaded_model_context(
            RuntimeDomain.RECOVERY, run_id
        )
        sdk_input = list(loaded.model_messages())
        assert sdk_input == [user]
        recorder = recorder_for(
            restored, 2, initial_messages=sdk_input, initial_context=loaded
        )
        capability = _AgentRunPersistenceCapability(
            recorder=recorder,
            agent_id="agent",
            agent_run_id=run_id,
            metadata={"agent_run_seq": "1"},
        )
        ctx = SimpleNamespace(
            messages=sdk_input, conversation_id=conversation, run_step=2
        )
        await capability.before_run(ctx)
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
        await capability.before_model_request(ctx, SimpleNamespace())
        received = []
        b = ModelResponse(
            parts=[
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
        assert recovered.messages == [user, a, user, response]
        assert recovered.context_messages == [user, response]
        commands = RuntimeStateCommands(
            repository,
            namespace=namespace,
            events=state.execution.events,
            tools=state.recovery.tools,
            background_tasks=set(),
        )
        bridge = RuntimeToolOperationBridge(
            state.recovery.tools,
            state.object_store(RuntimeDomain.RECOVERY),
            namespace=namespace,
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id=run_id,
            binding_digest=execution.binding_digest,
            owner="resumed",
            background_tasks=set(),
            payload_policy=PayloadPolicy(),
            terminal_commands=commands,
            producer_generation=2,
        )
        call = response.parts[1]
        effects = []
        for attempt in range(2):
            bridge = RuntimeToolOperationBridge(
                state.recovery.tools,
                state.object_store(RuntimeDomain.RECOVERY),
                namespace=namespace,
                tenant_id="tenant",
                execution_id="execution",
                agent_run_id=run_id,
                binding_digest=execution.binding_digest,
                owner=f"resumed-{attempt}",
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("published_state", (None, "RUNNING", "INTERRUPTED"))
async def test_recovery_backfill_is_atomic_but_does_not_invent_missing_associations(
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
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        root_store = state.run_store
        store = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=root_store.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=root_store.read_store(RuntimeDomain.EXECUTION),
            recovery_archive=root_store.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await store.initialize()
        public = store.read_store(RuntimeDomain.EXECUTION)
        recovery = store.read_store(RuntimeDomain.RECOVERY)
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
        user = ModelRequest(parts=[UserPromptPart("recoverable input")])
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
        await store.flush_execution_projection(
            run_id, execution_id="execution", producer_generation=1
        )
        # Use the existing recovery-only route to model a checkpoint that succeeds
        # before the delayed public batch, then discard this process-local staging.
        recovery_only = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=store.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=None,
            recovery_archive=recovery,
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await recovery_only.initialize()
        recorder = AgentRunRecorder(
            recovery_only, execution_id=None, agent_run_id=run_id
        )
        await recorder.register_agent_run(run)
        response = ModelResponse(parts=[TextPart("completed while public lagged")])
        model = TestModel()
        journal = ModelRequestJournal(
            source_namespace=namespace,
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id=run_id,
        )
        fact = journal.begin(1)
        recorder.begin_model_interaction(
            fact, model, [user], None, ModelRequestParameters(), False
        )
        recorder.prepare_model_interaction(
            fact, model, [user], None, ModelRequestParameters(), False
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
        await recorder.save_checkpoint(
            AgentRunCheckpoint(
                run_id,
                1,
                [user, response],
                context_messages=[user, response],
                transcript_message_count_before=1,
            )
        )
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [user]
        assert await public.model_interaction_count(agent_run_id=run_id) == 0
        assert await recovery.model_interaction_count(agent_run_id=run_id) == 1
        published_request = None
        if published_state is not None:
            source = await recovery.list_model_interactions(agent_run_id=run_id)
            relocated = await public.prepare_relocated_interactions(
                source, await recovery.resolve_model_interactions(source)
            )
            unfinished = replace(
                relocated[0],
                status=published_state,
                response_context=None,
                error_code=None,
                duration_ns=None,
                usage=None,
                finished_at=None,
            )
            published_request = unfinished.request_context
            await public.sync_projection(
                run,
                events=(),
                checkpoints=(),
                interactions=(unfinished,),
                execution_id="execution",
            )
        root_store.bind_execution_producer(
            run_id, execution_id="execution", producer_generation=1
        )
        if published_state == "INTERRUPTED":
            with pytest.raises(AIError) as conflict:
                await root_store.register_agent_run(run, execution_id="execution")
            assert conflict.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert (await public.list_model_interactions(agent_run_id=run_id))[
                0
            ].status == "INTERRUPTED"
            return
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
            await root_store.materialize_from_recovery(
                target=RuntimeDomain.EXECUTION,
                agent_run_id=run_id,
                execution_id="execution",
            )
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [user]
        assert await public.model_interaction_count(agent_run_id=run_id) == int(
            published_state is not None
        )
        if published_state is not None:
            assert (await public.list_model_interactions(agent_run_id=run_id))[
                0
            ].status == "RUNNING"
        fail[0] = False
        await root_store.register_agent_run(run, execution_id="execution")
        assert [m async for m in public.iter_messages(agent_run_id=run_id)] == [
            user,
            response,
        ]
        interactions = await public.list_model_interactions(agent_run_id=run_id)
        assert len(interactions) == 1 and interactions[0].model_request_seq == 1
        assert interactions[0].status == "SUCCEEDED"
        if published_request is not None:
            assert interactions[0].request_context == published_request
        resolved = await public.resolve_model_interactions(interactions)
        assert resolved[0][0] == (user,) and resolved[0][1] == (response,)
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_restored_completed_tool_group_consumes_published_pending_results(
    tmp_path: Path,
    backend: str,
) -> None:
    from linktools.ai.runtime._transcript_staging import StagedTranscript

    state = _storage(backend, tmp_path)
    await state.initialize(namespace="restored-tool-group", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        restored = state.run_store
        public = restored.read_store(RuntimeDomain.EXECUTION)
        recovery = restored.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run")
        user = ModelRequest(parts=[UserPromptPart("run tools")])
        response = ModelResponse(
            parts=[
                ToolCallPart("tool", {}, tool_call_id="quick"),
                ToolCallPart("tool", {}, tool_call_id="slow"),
            ]
        )
        quick = ToolReturnPart("tool", "quick result", tool_call_id="quick")
        slow = ToolReturnPart("tool", "slow result", tool_call_id="slow")
        completed = ModelRequest(parts=[quick, slow])
        before = AgentRunCheckpoint(
            "run", 1, [user, response], transcript_message_count_before=0
        )
        await public.sync_projection(
            run, events=(), checkpoints=(before,), execution_id="execution"
        )
        await recovery.materialize_checkpoint(run, before)
        pending = await public.transcript_repository.prepare_observation(
            "run",
            (),
            first_message_index=2,
            pending=ModelRequest(parts=[quick]),
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
        await recovery.materialize_checkpoint(
            run,
            AgentRunCheckpoint(
                "run",
                2,
                [user, response, completed],
                transcript_message_count_before=2,
            ),
        )
        restored.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1
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
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_required_checkpoint_rejects_replaced_producer_before_recovery_write(
    tmp_path: Path,
    backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="stale-recovery-checkpoint", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        root = state.run_store
        old = RuntimeAgentRunStore(
            StagingAgentRunStore(),
            conversation_archive=root.read_store(RuntimeDomain.CONVERSATION),
            execution_archive=root.read_store(RuntimeDomain.EXECUTION),
            recovery_archive=root.read_store(RuntimeDomain.RECOVERY),
            conversation_retention=RuntimeRetentionMode.VOLATILE,
            execution_retention=RuntimeRetentionMode.VOLATILE,
            recovery_retention=RuntimeRetentionMode.VOLATILE,
        )
        await old.initialize()
        old.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1
        )
        await old.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        updated = await repository.compare_and_swap(
            "execution",
            tenant_id="tenant",
            expected_revision=1,
            next_record=replace(execution, revision=2),
        )
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, updated)
        )
        with pytest.raises(AIError) as stale:
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
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert (
            await root.read_store(RuntimeDomain.RECOVERY).latest_checkpoint(
                agent_run_id="run"
            )
            is None
        )
        assert (
            await root.read_store(RuntimeDomain.EXECUTION).latest_checkpoint(
                agent_run_id="run"
            )
            is None
        )
    finally:
        await state.close()
