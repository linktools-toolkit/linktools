#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interrupted tool evidence does not replace a recovered terminal result."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import ToolDefinition

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._capabilities import _AgentRunPersistenceCapability
from linktools.ai.runtime._journal import DURATION_NS_METADATA_KEY, MODEL_REQUEST_SEQ_METADATA_KEY
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
)
from linktools.ai.runtime.state._step_contracts import (
    TOOL_ERROR_CODE_METADATA_KEY,
    AgentRunRecord,
    StepEvent,
)
from linktools.ai.runtime.state._steps import StagingAgentRunStore

from .test_history_request_association import _history


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", (False, True), ids=("completed", "pending"))
@pytest.mark.parametrize("outcome", ("success", "failed", "retry"))
async def test_recovered_tool_retains_interruptions_and_one_real_result(
    pending: bool, outcome: str,
) -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        fact = journal.begin(4)
        call = ToolCallPart("echo", {}, tool_call_id="call")
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
        )
        await recorder.record_tool_start(call, 4)
        placeholders = []
        for attempt in range(2):
            await recorder.record_event(
                "TOOL_CALL_FAILED", 4, tool_call_id="call", tool_name="echo",
                error=f"unknown attempt {attempt}",
                metadata={
                    TOOL_ERROR_CODE_METADATA_KEY: ErrorCode.TOOL_EFFECT_UNKNOWN.value,
                    MODEL_REQUEST_SEQ_METADATA_KEY: "1",
                },
            )
            placeholder = ToolReturnPart(
                "echo", f"interrupted attempt {attempt}", tool_call_id="call",
                outcome="interrupted", metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
            )
            placeholders.append(placeholder)
            await recorder.record_tool_result_boundary(placeholder, 4)
            if not pending:
                recorder.finish_transcript(interrupted=True)
            await recorder.commit_history_boundary()
            recorder = AgentRunRecorder(
                history.state.run_store, execution_id="execution",
                agent_run_id=recorder.agent_run_id,
                history_boundary=partial(
                    history.state.run_store.flush_execution_projection,
                    recorder.agent_run_id, execution_id="execution",
                ),
            )
            await recorder.register_agent_run(history.records[0])

        result = (
            RetryPromptPart("retry result", tool_name="echo", tool_call_id="call")
            if outcome == "retry" else ToolReturnPart(
                "echo", "confirmed result", tool_call_id="call", outcome=outcome,
            )
        )
        await recorder.record_tool_result_boundary(result, 5)
        recorder.append_transcript_message(ModelRequest(parts=[result]))
        history.recorders = (recorder,)
        assert [
            part for message in recorder.transcript_messages() for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        ] == [*placeholders, result]
        expected_status = "SUCCEEDED" if outcome == "success" else "FAILED"
        expected_kind = "tool_result" if outcome != "retry" else "retry"
        for archived in (False, True):
            if archived:
                await history.archive()
            page = await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=20, tool_call_id="call",
            )
            assert [(item.message_seq, item.item_kind, item.status) for item in page.items] == [
                (1, "tool_call", expected_status),
                (2, "tool_result", "INTERRUPTED"),
                (3, "tool_result", "INTERRUPTED"),
                (4, expected_kind, expected_status),
            ]
            assert {item.model_request_seq for item in page.items} == {1}
            assert {item.step_index for item in page.items} == {4}
            trace = await history.reader.trace(
                "execution", tenant_id="tenant", cursor=None, limit=20, tool_call_id="call",
            )
            assert [item.payload["status"] for item in trace.items] == [
                "STARTED", "EFFECT_UNKNOWN", "EFFECT_UNKNOWN", expected_status,
            ]
            assert all(
                item.payload["error_code"] == ErrorCode.TOOL_EFFECT_UNKNOWN.value
                for item in trace.items[1:3]
            )

        archive = history.state.run_store.read_store(RuntimeDomain.EXECUTION)
        events = await archive.list_events(agent_run_id=recorder.agent_run_id)
        assert [event.error for event in events if event.error is not None] == [
            "unknown attempt 0", "unknown attempt 1",
        ]
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = tuple(sorted(
            await archive.state_store.read(lambda tx: tx.scan_facts()),
            key=lambda fact: (fact.stream_digest, fact.sequence),
        ))
        aliases, sequences = canonical_snapshot_indexes(
            namespace="history", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
            records=records, facts=facts, operations=(),
        )
        validate_snapshot_domain(
            namespace="history", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
            records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
        )

        restored = AgentRunRecorder(
            history.state.run_store, execution_id="execution", agent_run_id=recorder.agent_run_id,
        )
        await restored.register_agent_run(history.records[0])
        before = restored.transcript_messages()
        await restored.record_tool_result_boundary(replace(result), 6)
        restored.append_transcript_message(ModelRequest(parts=[replace(result)]))
        assert restored.transcript_messages() == before
        with pytest.raises(AIError) as raised:
            await restored.record_tool_result_boundary(replace(result, content="different result"), 6)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "metadata"),
    (
        ("interrupted", None),
        ("interrupted", "tool metadata"),
        ("interrupted", {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: "true"}),
        ("success", {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}),
    ),
)
async def test_only_explicit_synthesized_interruptions_allow_a_later_result(
    outcome: str, metadata: object,
) -> None:
    store = StagingAgentRunStore()
    run = AgentRunRecord("run")
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await recorder.register_agent_run(run)
    original = ToolReturnPart(
        "echo", "unconfirmed text", tool_call_id="call", outcome=outcome,
        metadata=metadata,
    )
    recorder.append_transcript_message(ModelRequest(parts=[original]))
    restored = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await restored.register_agent_run(run)
    changed = replace(original, content="different result")
    with pytest.raises(AIError) as raised:
        await restored.record_tool_result_boundary(changed, 1)
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    with pytest.raises(AIError) as raised:
        restored.append_transcript_message(ModelRequest(parts=[changed]))
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
async def test_request_append_keeps_pending_interruption_before_actual_result() -> None:
    store = StagingAgentRunStore()
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await recorder.register_agent_run(AgentRunRecord("run"))
    placeholder = ToolReturnPart(
        "echo", "interrupted", tool_call_id="call", outcome="interrupted",
        metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
    )
    recorder.stage_tool_result(placeholder)
    result = ToolReturnPart("echo", "result", tool_call_id="call")
    recorder.append_transcript_message(ModelRequest(parts=[result]))
    await recorder.record_tool_result_boundary(result, 1)
    messages = recorder.transcript_messages()
    assert messages[0].state == "interrupted"
    assert [message.parts for message in messages] == [[placeholder], [result]]
    assert [event.event_type for event in await store.list_events(agent_run_id="run")] == [
        "TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", (AIError(ErrorCode.TOOL_EFFECT_UNKNOWN), RuntimeError("TOOL_EFFECT_UNKNOWN")),
    ids=("typed-unknown", "untyped-error"),
)
async def test_only_typed_unknown_effect_errors_are_unfinished_attempts(error: Exception) -> None:
    store = StagingAgentRunStore()
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    await recorder.register_agent_run(AgentRunRecord("run"))
    capability = _AgentRunPersistenceCapability(
        recorder=recorder, agent_id="agent", agent_run_id="run",
    )
    call = ToolCallPart("echo", {}, tool_call_id="call")
    definition = ToolDefinition(name="echo")
    context = SimpleNamespace(run_step=1)
    await capability.before_tool_execute(context, call=call, tool_def=definition, args={})
    with pytest.raises(type(error)):
        await capability.on_tool_execute_error(
            context, call=call, tool_def=definition, args={}, error=error,
        )
    events = await store.list_events(agent_run_id="run")
    assert events[-1].event_type == "TOOL_CALL_FAILED"
    assert events[-1].error == repr(error)
    assert events[-1].metadata.get(TOOL_ERROR_CODE_METADATA_KEY) == (
        ErrorCode.TOOL_EFFECT_UNKNOWN.value if isinstance(error, AIError) else None
    )
    await recorder.record_tool_result_boundary(ToolReturnPart("echo", "result", tool_call_id="call"), 1)
    events = await store.list_events(agent_run_id="run")
    assert [event.event_type for event in events] == [
        "TOOL_CALL_STARTED", "TOOL_CALL_FAILED",
        *(["TOOL_CALL_SUCCEEDED"] if isinstance(error, AIError) else []),
    ]


@pytest.mark.asyncio
async def test_pending_placeholder_cursor_keeps_selected_keys_after_request_reorders() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        fact = journal.begin(1)
        calls = [ToolCallPart("echo", {}, tool_call_id=value) for value in ("call", "ordinary", "late")]
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=calls), include_observation=False,
        )
        for call in calls:
            await recorder.record_tool_start(call, 1)
        ordinary = ToolReturnPart("echo", "ordinary result", tool_call_id="ordinary")
        placeholder = ToolReturnPart(
            "echo", "interrupted result", tool_call_id="call", outcome="interrupted",
            metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
        )
        await recorder.record_tool_result_boundary(ordinary, 1)
        await recorder.record_tool_result_boundary(placeholder, 1)
        first = await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=1)
        assert first.next_cursor is not None
        late = ToolReturnPart("echo", "later result", tool_call_id="late")
        await recorder.record_tool_result_boundary(late, 1)
        recorder.append_transcript_message(ModelRequest(
            parts=[placeholder, ordinary, late], instructions="next request instructions",
        ))
        await recorder.commit_history_boundary()
        items = list(first.items)
        cursor = first.next_cursor
        while cursor is not None:
            page = await history.reader.history("execution", tenant_id="tenant", cursor=cursor, limit=1)
            items.extend(page.items)
            cursor = page.next_cursor
        results = [item for item in items if item.item_kind == "tool_result"]
        assert [(item.tool_call_id, item.status, item.message_seq, item.part_index) for item in results] == [
            ("ordinary", "SUCCEEDED", 2, None), ("call", "INTERRUPTED", 2, None),
        ]
        assert not any(item.content == "next request instructions" for item in items)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("call", "request", "duration", "conflicting-request"))
async def test_unknown_attempts_still_validate_tool_identity_and_coordinates(corruption: str) -> None:
    async with _history() as history:
        metadata = {
            TOOL_ERROR_CODE_METADATA_KEY: ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            MODEL_REQUEST_SEQ_METADATA_KEY: "1",
        }
        if corruption == "request":
            metadata[MODEL_REQUEST_SEQ_METADATA_KEY] = "invalid"
        elif corruption == "duration":
            metadata[DURATION_NS_METADATA_KEY] = "invalid"
        elif corruption == "conflicting-request":
            metadata[MODEL_REQUEST_SEQ_METADATA_KEY] = "2"
        events = [StepEvent(
            "run", "TOOL_CALL_STARTED", 1, tool_call_id="call",
            metadata={MODEL_REQUEST_SEQ_METADATA_KEY: "1"},
        ), StepEvent(
            "run", "TOOL_CALL_FAILED", 1,
            tool_call_id="" if corruption == "call" else "call", metadata=metadata,
        )]
        with pytest.raises(AIError) as raised:
            await history.reader._tool_call_metadata(
                "run", execution_id="execution", tenant_id="tenant", events=events,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
