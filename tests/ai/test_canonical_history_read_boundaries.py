#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public cursors bind membership and associations to committed read cuts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import HmacCursorSigner, agent_conversation_id, agent_run_id
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._runtime_history import _merge_usage_summaries
from linktools.ai.runtime.service_api import UsageSummary
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import ExecutionHistoryHeadRecord, ExecutionHistoryState
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent

from .test_model_interaction_lifecycle_paging import _Executions, _HistoryStore, _interaction, _record


@pytest.mark.asyncio
async def test_usage_retains_running_and_interrupted_unknowns_without_fake_cancellation() -> None:
    execution = _record("root")
    execution.binding_kind = "agent"
    run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=1)
    store = _HistoryStore()
    store.runs[run_id] = AgentRunRecord(
        run_id, agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root"),
        metadata={"agent_run_seq": "1"},
    )
    store.interactions[run_id] = [replace(
        _interaction(run_id, sequence), status=status, duration_ns=None, finished_at=None,
        request_context=None, request_envelope=None,
    ) for sequence, status in enumerate(("RUNNING", "INTERRUPTED"), 1)]
    reader = StepExecutionHistoryReader(
        namespace="history", executions=_Executions(execution), store=store,
        cursor_signer=HmacCursorSigner("history", b"history"),
    )
    result = await reader.usage("root", tenant_id="tenant")
    assert result.logical_requests == 2
    assert result.running_requests == result.interrupted_requests == 1
    assert result.cancelled_requests == result.failed_requests == result.succeeded_requests == 0
    assert result.unknown_usage_requests == result.unknown_duration_requests == 2
    assert result.input_tokens == result.output_tokens == result.model_duration_ns == 0
    merged = _merge_usage_summaries([result, UsageSummary(logical_requests=1, running_requests=1,
                                                       unknown_usage_requests=1, unknown_duration_requests=1)])
    assert merged.logical_requests == 3 and merged.running_requests == 2
    assert merged.interrupted_requests == 1 and merged.cancelled_requests == 0


@pytest.mark.asyncio
async def test_trace_index_preserves_timestamp_order_and_fixed_event_membership(tmp_path: Path) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "history.db")
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = storage.execution.executions
        await executions.state_store.mutate(lambda transaction: executions.insert_history_head_in_transaction(
            transaction, ExecutionHistoryHeadRecord("root", ExecutionHistoryState.OPEN, 0, None),
        ))
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=1)
        conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
        run = AgentRunRecord(run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": "1"})
        await archive.register_agent_run(run, execution_id="root")
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for sequence, seconds in ((1, 2), (2, 1)):
            await archive.append_event(StepEvent(
                run_id, "MODEL_REQUEST_STARTED", sequence, timestamp=started + timedelta(seconds=seconds),
                agent_conversation_id=conversation_id,
                metadata={"linktools.ai.model_request_seq": str(sequence)},
            ), execution_id="root")
        reader = StepExecutionHistoryReader(
            namespace="history", executions=_Executions(_record("root")), store=archive,
            cursor_signer=HmacCursorSigner("history", b"history"),
        )
        first = await reader.trace("root", tenant_id="tenant", cursor=None, limit=1)
        assert [value.payload["model_request_seq"] for value in first.items] == [2]
        assert first.next_cursor is not None
        await archive.append_event(StepEvent(
            run_id, "MODEL_REQUEST_STARTED", 3, timestamp=started,
            agent_conversation_id=conversation_id, metadata={"linktools.ai.model_request_seq": "3"},
        ), execution_id="root")
        tail = await reader.trace("root", tenant_id="tenant", cursor=first.next_cursor, limit=1)
        assert [value.payload["model_request_seq"] for value in tail.items] == [1]
        assert tail.next_cursor is None
        fresh = await reader.trace("root", tenant_id="tenant", cursor=None, limit=10)
        assert [value.payload["model_request_seq"] for value in fresh.items] == [3, 2, 1]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_history_cursor_preserves_response_association_cutoff(tmp_path: Path) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "associations.db")
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = storage.execution.executions
        await executions.state_store.mutate(lambda transaction: executions.insert_history_head_in_transaction(
            transaction, ExecutionHistoryHeadRecord("root", ExecutionHistoryState.OPEN, 0, None),
        ))
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=1)
        conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
        run = AgentRunRecord(run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": "1"})
        await archive.register_agent_run(run, execution_id="root")
        repository = archive.transcript_repository
        prepared = await repository.prepare_observation(
            run_id, (ModelRequest(parts=[UserPromptPart("question")]), ModelResponse(parts=[TextPart("answer")])),
            first_message_index=0, pending=None, pending_keys=(),
        )
        await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, prepared))
        await archive.append_event(StepEvent(
            run_id, "MODEL_REQUEST_STARTED", 4, agent_conversation_id=conversation_id,
            metadata={"linktools.ai.model_request_seq": "1"},
        ), execution_id="root")
        reader = StepExecutionHistoryReader(
            namespace="history", executions=_Executions(_record("root")), store=archive,
            cursor_signer=HmacCursorSigner("history", b"history"),
        )
        first = await reader.history("root", tenant_id="tenant", cursor=None, limit=1)
        assert first.next_cursor is not None
        await archive.append_event(StepEvent(
            run_id, "MODEL_REQUEST_SUCCEEDED", 4, agent_conversation_id=conversation_id,
            metadata={"linktools.ai.model_request_seq": "1", "linktools.ai.message_seq": "2"},
        ), execution_id="root")
        tail = await reader.history("root", tenant_id="tenant", cursor=first.next_cursor, limit=1)
        assert len(tail.items) == 1 and tail.items[0].content == "answer"
        assert tail.items[0].model_request_seq is None and tail.items[0].step_index is None
        fresh = await reader.history("root", tenant_id="tenant", cursor=None, limit=10)
        assert fresh.items[-1].model_request_seq == 1 and fresh.items[-1].step_index == 4
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_history_cursor_keeps_an_empty_captured_run_empty_after_completion(tmp_path: Path) -> None:
    from linktools.ai.core import ExecutionStatus

    storage = RuntimeStorage.sqlite(tmp_path / "empty-run.db")
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = storage.execution.executions
        await executions.state_store.mutate(lambda transaction: executions.insert_history_head_in_transaction(
            transaction, ExecutionHistoryHeadRecord("root", ExecutionHistoryState.OPEN, 0, None),
        ))
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
        run_ids = tuple(agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=sequence)
                        for sequence in (1, 2))
        for sequence, run_id in enumerate(run_ids, 1):
            await archive.register_agent_run(AgentRunRecord(
                run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": str(sequence)},
            ), execution_id="root")
        repository = archive.transcript_repository
        first_run = await repository.prepare_observation(
            run_ids[0], (ModelRequest(parts=[UserPromptPart("first"), UserPromptPart("second")]),),
            first_message_index=0, pending=None, pending_keys=(),
        )
        await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, first_run))
        execution = _record("root")
        execution.agent_run_seq = 2
        reader = StepExecutionHistoryReader(
            namespace="history", executions=_Executions(execution), store=archive,
            cursor_signer=HmacCursorSigner("history", b"history"),
        )
        first = await reader.history("root", tenant_id="tenant", cursor=None, limit=1)
        assert first.next_cursor is not None
        second_run = await repository.prepare_observation(
            run_ids[1], (ModelResponse(parts=[TextPart("later run")]),),
            first_message_index=0, pending=None, pending_keys=(),
        )
        await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, second_run))
        execution.status = ExecutionStatus.SUCCEEDED
        tail = await reader.history("root", tenant_id="tenant", cursor=first.next_cursor, limit=100)
        assert [(value.agent_run_seq, value.content) for value in tail.items] == [(1, "second")]
        assert tail.next_cursor is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_history_and_trace_pagination_preserve_cross_run_and_child_order(tmp_path: Path) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "source-order.db")
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = storage.execution.executions
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        root = _record("root")
        root.agent_run_seq = 2
        child = _record("child", child=True)
        fake = _Executions(root)
        fake.children.append(child)
        for execution in (root, child):
            await executions.state_store.mutate(lambda transaction: executions.insert_history_head_in_transaction(
                transaction, ExecutionHistoryHeadRecord(execution.execution_id, ExecutionHistoryState.OPEN, 0, None),
            ))
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for execution_id, sequence, times in (("root", 1, (2, 1)), ("root", 2, (1, 2)), ("child", 1, (1, 2))):
            run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id=execution_id, agent_run_seq=sequence)
            conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id=execution_id)
            await archive.register_agent_run(AgentRunRecord(
                run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": str(sequence)},
            ), execution_id=execution_id)
            repository = archive.transcript_repository
            prepared = await repository.prepare_observation(
                run_id, (ModelRequest(parts=[UserPromptPart(f"{execution_id}:{sequence}:a"),
                                             UserPromptPart(f"{execution_id}:{sequence}:b")]),),
                first_message_index=0, pending=None, pending_keys=(),
            )
            await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, prepared))
            for request, seconds in enumerate(times, 1):
                await archive.append_event(StepEvent(
                    run_id, "MODEL_REQUEST_STARTED", request,
                    timestamp=started + timedelta(seconds=seconds), agent_conversation_id=conversation_id,
                    metadata={"linktools.ai.model_request_seq": str(request)},
                ), execution_id=execution_id)
        reader = StepExecutionHistoryReader(
            namespace="history", executions=fake, store=archive,
            cursor_signer=HmacCursorSigner("history", b"history"),
        )
        history_items, trace_items = [], []
        for method, target in ((reader.history, history_items), (reader.trace, trace_items)):
            cursor = None
            while True:
                page = await method("root", tenant_id="tenant", cursor=cursor, limit=1)
                target.extend(page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
        assert [value.content for value in history_items] == [
            "root:1:a", "root:1:b", "root:2:a", "root:2:b", "child:1:a", "child:1:b",
        ]
        assert [(value.execution_id, value.payload["agent_run_seq"], value.step_event_seq)
                for value in trace_items] == [
                    ("root", 1, 2), ("root", 2, 1), ("child", 1, 1),
                    ("root", 1, 1), ("root", 2, 2), ("child", 1, 2),
                ]
        filtered = await reader.trace("root", tenant_id="tenant", cursor=None, limit=1, model_request_seq=2)
        remaining = await reader.trace("root", tenant_id="tenant", cursor=filtered.next_cursor,
                                       limit=10, model_request_seq=2)
        assert [(value.execution_id, value.payload["agent_run_seq"], value.step_event_seq)
                for value in (*filtered.items, *remaining.items)] == [
                    ("root", 1, 2), ("root", 2, 2), ("child", 1, 2),
                ]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_history_replay_projection_omits_only_duplicate_calls_at_original_coordinates(tmp_path: Path) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart

    storage = RuntimeStorage.sqlite(tmp_path / "replayed-calls.db")
    await storage.initialize(namespace="history", tenant_id="tenant")
    try:
        executions = storage.execution.executions
        await executions.state_store.mutate(lambda transaction: executions.insert_history_head_in_transaction(
            transaction, ExecutionHistoryHeadRecord("root", ExecutionHistoryState.OPEN, 0, None),
        ))
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=1)
        conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
        await archive.register_agent_run(AgentRunRecord(
            run_id, agent_conversation_id=conversation_id, metadata={"agent_run_seq": "1"},
        ), execution_id="root")
        original = ToolCallPart("echo", {"value": "original"}, tool_call_id="replayed")
        repeated = ModelResponse(parts=[
            original, TextPart("new text"), ToolCallPart("echo", {"value": "new"}, tool_call_id="new"),
        ])
        messages = (
            ModelRequest(parts=[UserPromptPart("question")]), ModelResponse(parts=[original]),
            ModelRequest(parts=[ToolReturnPart("echo", "original", tool_call_id="replayed")]), repeated,
        )
        repository = archive.transcript_repository
        prepared = await repository.prepare_observation(
            run_id, messages[:3], first_message_index=0, pending=repeated,
            pending_keys=("part:0", "part:1", "part:2"),
        )
        await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, prepared))
        async def append_request(request: int, message: int, call_id: str) -> None:
            await archive.append_event(StepEvent(
                run_id, "MODEL_REQUEST_SUCCEEDED", request, agent_conversation_id=conversation_id,
                metadata={
                    "linktools.ai.model_request_seq": str(request), "linktools.ai.message_seq": str(message),
                    **({"linktools.ai.replayed_tool_call_indices": "[0]"} if request == 2 else {}),
                },
            ), execution_id="root")
            await archive.append_event(StepEvent(
                run_id, "TOOL_CALL_STARTED", request, tool_name="echo", tool_call_id=call_id,
                agent_conversation_id=conversation_id,
                metadata={"linktools.ai.model_request_seq": str(request)},
            ), execution_id="root")
        await append_request(1, 2, "replayed")
        reader = StepExecutionHistoryReader(
            namespace="history", executions=_Executions(_record("root")), store=archive,
            cursor_signer=HmacCursorSigner("history", b"history"),
        )
        pending = await reader.history("root", tenant_id="tenant", cursor=None, limit=100)
        frozen = await reader.history("root", tenant_id="tenant", cursor=None, limit=1)
        assert [(item.item_kind, item.part_index) for item in pending.items if item.message_seq == 4] == [
            ("assistant", 1), ("tool_call", 2),
        ]
        assert all(item.model_request_seq is None for item in pending.items if item.message_seq == 4)
        completed = await repository.prepare_observation(
            run_id, (repeated,), first_message_index=3, pending=None, pending_keys=(),
        )
        await archive.state_store.mutate(lambda transaction: repository.commit_observation(transaction, completed))
        await append_request(2, 4, "new")
        frozen_items = list(frozen.items)
        cursor = frozen.next_cursor
        while cursor is not None:
            page = await reader.history("root", tenant_id="tenant", cursor=cursor, limit=1)
            frozen_items.extend(page.items)
            cursor = page.next_cursor
        assert tuple(frozen_items) == pending.items
        items = []
        cursor = None
        while True:
            page = await reader.history("root", tenant_id="tenant", cursor=cursor, limit=1)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert [(item.item_kind, item.message_seq, item.part_index, item.model_request_seq) for item in items] == [
            ("user", 1, 0, None), ("tool_call", 2, 0, 1), ("tool_result", 3, None, 1),
            ("assistant", 4, 1, 2), ("tool_call", 4, 2, 2),
        ]
        assert (await reader.history("root", tenant_id="tenant", cursor=None, limit=10,
                                     message_seq=4, part_index=0)).items == ()
        selected = await reader.history("root", tenant_id="tenant", cursor=None, limit=10,
                                        message_seq=4, part_index=2)
        assert selected.items == (items[-1],)
        stored = [message async for message in archive.iter_message_range(agent_run_id=run_id, start=3, end=4)]
        assert stored == [repeated]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_indices", ("[true]", "[0]"))
async def test_history_rejects_invalid_replay_metadata_instead_of_hiding_content(replay_indices: str) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    from .test_history_request_association import _history

    async with _history() as history:
        recorder, = history.recorders
        recorder.append_transcript_message(ModelResponse(parts=[TextPart("visible text")]))
        await recorder.record_event("MODEL_REQUEST_SUCCEEDED", 1, metadata={
            "linktools.ai.model_request_seq": "1", "linktools.ai.message_seq": "1",
            "linktools.ai.replayed_tool_call_indices": replay_indices,
        })
        await recorder.commit_history_boundary()
        with pytest.raises(AIError) as invalid:
            await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=10)
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
