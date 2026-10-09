#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw history retains proven model-request origins across durable boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

import pytest
from pydantic_ai import Agent, ModelRequestNode, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    agent_conversation_id,
    agent_run_id,
)
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._capabilities import _AgentRunPersistenceCapability
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.runtime.state._steps import StagingAgentRunStore

from .test_history_projection_conformance import _record


@dataclass
class _History:
    state: RuntimeStorage
    reader: StepExecutionHistoryReader
    recorders: tuple[AgentRunRecorder, ...]
    journals: tuple[ModelRequestJournal, ...]
    records: tuple[AgentRunRecord, ...]

    async def archive(self) -> None:
        for recorder in self.recorders:
            recorder.finish_transcript()
            previous = await recorder.latest_checkpoint(include_interrupted=True)
            await recorder.save_checkpoint(
                AgentRunCheckpoint(
                    recorder.agent_run_id,
                    10,
                    list(recorder.transcript_messages()),
                    transcript_message_count_before=0 if previous is None else len(previous.messages),
                )
            )
            await self.state.run_store.flush_execution_projection(
                recorder.agent_run_id, execution_id="execution"
            )
        await self.state.run_store.release_staging_many(
            candidate_agent_run_ids=tuple(r.agent_run_id for r in self.recorders),
            execution_id="execution",
        )


@asynccontextmanager
async def _history(run_count: int = 1) -> AsyncIterator[_History]:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="history", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(
            _record(ExecutionStatus.STARTED, run_count)
        )
        reader = StepExecutionHistoryReader(
            namespace="history",
            executions=state.execution.executions,
            store=state.run_store.read_store(RuntimeDomain.EXECUTION),
            notifications=state.run_store,
            cursor_signer=HmacCursorSigner("history", b"history-key"),
        )
        recorders = []
        journals = []
        records = []
        for sequence in range(1, run_count + 1):
            run_id = agent_run_id(
                namespace="history", tenant_id="tenant", execution_id="execution",
                agent_run_seq=sequence,
            )
            recorder = AgentRunRecorder(
                state.run_store, execution_id="execution", agent_run_id=run_id,
                history_boundary=partial(state.run_store.flush_execution_projection, run_id, execution_id="execution"),
            )
            record = AgentRunRecord(
                run_id,
                agent_conversation_id=agent_conversation_id(
                    namespace="history", tenant_id="tenant", execution_id="execution"
                ),
                agent_id="default",
                metadata={"agent_run_seq": str(sequence), "agent_id": "default"},
                started_at=datetime.now(timezone.utc),
            )
            await recorder.register_agent_run(record)
            recorders.append(recorder)
            records.append(record)
            journals.append(ModelRequestJournal(
                source_namespace="history", tenant_id="tenant", execution_id="execution",
                agent_run_id=run_id,
            ))
        yield _History(state, reader, tuple(recorders), tuple(journals), tuple(records))
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_history_response_and_tool_result_keep_origin_after_archive() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        recorder.append_transcript_message(ModelRequest(parts=[
            SystemPromptPart("system input"), UserPromptPart("user input"),
        ]))
        fact = journal.begin(3)
        await recorder.record_model_event(fact, phase="started", include_observation=False)
        call = ToolCallPart("echo", {"value": "answer"}, tool_call_id="call")
        response = ModelResponse(parts=[ThinkingPart("reason"), TextPart("answer"), call])
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=response, include_observation=False,
        )
        result = ToolReturnPart("echo", None, tool_call_id="call")
        recorder.stage_tool_result(result)
        await recorder.record_tool_result_boundary(result, 4)
        recorder.append_transcript_message(ModelRequest(parts=[
            result, RetryPromptPart("validate the output"),
        ]))
        await recorder.commit_history_boundary()
        live = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100
        )
        assert [item.model_request_seq for item in live.items] == [None, None, 1, 1, 1, 1, None]
        assert [item.step_index for item in live.items] == [None, None, 3, 3, 3, 3, None]
        assert [item.message_seq for item in live.items] == [1, 1, 2, 2, 2, 3, 3]
        assert [item.part_index for item in live.items[:5]] == [0, 1, 0, 1, 2]
        assert live.items[5].content is None
        assert live.items[5].content_included is True
        assert recorder.transcript_messages()[1] == response

        await history.archive()
        archived = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100
        )
        assert archived.items == live.items
        selected = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100,
            model_request_seq=1, step_index=3,
        )
        assert selected.items == live.items[2:6]
        assistant = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100,
            message_seq=2, part_index=1,
        )
        assert assistant.items == (live.items[3],)
        wrong_step = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, step_index=4
        )
        assert wrong_step.items == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("preceding_request", ["compaction", "failed", "output_retry"])
async def test_history_same_step_requests_use_actual_response_slots(
    preceding_request: str,
) -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart("question")]))
        first = journal.begin(5, purpose="compaction" if preceding_request == "compaction" else "agent")
        await recorder.record_model_event(first, phase="started", include_observation=False)
        failed = preceding_request == "failed"
        await recorder.record_model_event(
            journal.finish(first.model_request_seq, status="FAILED" if failed else "SUCCEEDED"),
            phase="failed" if failed else "completed",
            response=None if failed else ModelResponse(parts=[TextPart("first answer")]),
            error_code="MODEL_REQUEST_FAILED" if failed else None,
            include_observation=False,
        )
        if preceding_request == "output_retry":
            recorder.append_transcript_message(ModelRequest(parts=[RetryPromptPart("try again")]))
        else:
            assert len(recorder.transcript_messages()) == 1
        second = journal.begin(5, output_retry_index=1 if preceding_request == "output_retry" else None)
        await recorder.record_model_event(second, phase="started", include_observation=False)
        await recorder.record_model_event(
            journal.finish(second.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[TextPart("accepted answer")]),
            include_observation=False,
        )
        expected_slot = 4 if preceding_request == "output_retry" else 2
        for archived in (False, True):
            if archived:
                await history.archive()
            page = await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=100, model_request_seq=2,
            )
            assert [(item.message_seq, item.content, item.model_request_seq, item.step_index)
                    for item in page.items] == [(expected_slot, "accepted answer", 2, 5)]
            first_page = await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=100, model_request_seq=1,
            )
            assert [item.content for item in first_page.items] == (
                ["first answer"] if preceding_request == "output_retry" else []
            )


@pytest.mark.asyncio
async def test_history_cursor_does_not_gain_unfinished_response_after_completion() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        recorder.append_transcript_message(ModelRequest(parts=[
            SystemPromptPart("instructions"), UserPromptPart("question"),
        ]))
        fact = journal.begin(2)
        await recorder.record_model_event(fact, phase="started", include_observation=False)
        parts = [ThinkingPart("reason"), TextPart("answer")]
        for index, part in enumerate(parts):
            recorder.stage_response_part(part, index)
        await recorder.commit_history_boundary()
        before = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=1,
        )
        assert before.next_cursor is not None
        partial = await history.reader.history(
            "execution", tenant_id="tenant", cursor=before.next_cursor, limit=100,
        )
        assert [item.content for item in partial.items] == ["question"]
        assert all(item.model_request_seq is None and item.step_index is None for item in partial.items)
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=parts), include_observation=False,
        )
        await history.archive()
        continuation = await history.reader.history(
            "execution", tenant_id="tenant", cursor=before.next_cursor, limit=100,
        )
        assert continuation.items == partial.items
        fresh = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, model_request_seq=1,
        )
        assert [item.content for item in fresh.items] == ["reason", "answer"]
        assert all(item.model_request_seq == 1 and item.step_index == 2 for item in fresh.items)


@pytest.mark.asyncio
async def test_history_reused_tool_call_id_keeps_each_runs_request() -> None:
    async with _history(run_count=2) as history:
        for run_sequence, (recorder, journal) in enumerate(
            zip(history.recorders, history.journals), start=1
        ):
            if run_sequence == 2:
                failed = journal.begin(1)
                await recorder.record_model_event(
                    journal.finish(failed.model_request_seq, status="FAILED"),
                    phase="failed", error_code="MODEL_REQUEST_FAILED", include_observation=False,
                )
            fact = journal.begin(run_sequence + 2)
            call = ToolCallPart("echo", {"run": run_sequence}, tool_call_id="same-call")
            await recorder.record_model_event(
                journal.finish(fact.model_request_seq, status="SUCCEEDED"),
                phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
            )
            result = ToolReturnPart("echo", f"run {run_sequence}", tool_call_id="same-call")
            recorder.stage_tool_result(result)
            await recorder.record_tool_result_boundary(result, 9)
        for archived in (False, True):
            if archived:
                await history.archive()
            for run_sequence in (1, 2):
                page = await history.reader.history(
                    "execution", tenant_id="tenant", cursor=None, limit=100,
                    agent_run_seq=run_sequence, tool_call_id="same-call",
                )
                assert [item.content for item in page.items] == [{"run": run_sequence}, f"run {run_sequence}"]
                assert {item.model_request_seq for item in page.items} == {run_sequence}
                assert {item.step_index for item in page.items} == {run_sequence + 2}


@pytest.mark.asyncio
async def test_history_deferred_result_restores_same_run_request_after_recovery() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        fact = journal.begin(6)
        call = ToolCallPart("echo", {}, tool_call_id="deferred-call")
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
        )
        await recorder.save_checkpoint(AgentRunCheckpoint(
            recorder.agent_run_id, 6, list(recorder.transcript_messages()),
            state="interrupted", transcript_message_count_before=0,
        ))
        await history.state.run_store.flush_execution_projection(
            recorder.agent_run_id, execution_id="execution"
        )
        await history.state.run_store.release_staging_many(
            candidate_agent_run_ids=(recorder.agent_run_id,), execution_id="execution"
        )
        recovered = AgentRunRecorder(
            history.state.run_store, execution_id="execution", agent_run_id=recorder.agent_run_id,
            history_boundary=partial(history.state.run_store.flush_execution_projection,
                                     recorder.agent_run_id, execution_id="execution"),
        )
        await recovered.register_agent_run(history.records[0])
        result = ToolReturnPart("echo", "resumed result", tool_call_id="deferred-call")
        recovered.stage_tool_result(result)
        await recovered.record_tool_result_boundary(result, 7)
        history.recorders = (recovered,)
        for archived in (False, True):
            if archived:
                await history.archive()
            page = await history.reader.history(
                "execution", tenant_id="tenant", cursor=None, limit=100, model_request_seq=1,
            )
            assert [item.item_kind for item in page.items] == ["tool_call", "tool_result"]
            assert {item.step_index for item in page.items} == {6}
            assert page.items[1].content == "resumed result"


@pytest.mark.asyncio
async def test_history_carried_tool_result_does_not_borrow_new_runs_request() -> None:
    async with _history(run_count=2) as history:
        original, resumed = history.recorders
        first, second = history.journals
        call = ToolCallPart("echo", {}, tool_call_id="carried-call")
        fact = first.begin(3)
        await original.record_model_event(
            first.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
        )
        carried = ToolReturnPart("echo", "previous run result", tool_call_id="carried-call")
        resumed.append_transcript_message(ModelRequest(parts=[carried]))
        await resumed.record_tool_result_boundary(carried, 3)
        fact = second.begin(3)
        await resumed.record_model_event(
            second.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[TextPart("new run answer")]),
            include_observation=False,
        )
        await history.archive()
        page = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=100, agent_run_seq=2,
        )
        assert [(item.content, item.model_request_seq, item.step_index) for item in page.items] == [
            ("previous run result", None, None), ("new run answer", 1, 3),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_node", [False, True])
async def test_sdk_after_node_boundary_retains_one_actual_raw_response(
    fail_after_node: bool,
) -> None:
    store = StagingAgentRunStore()
    recorder = AgentRunRecorder(store, execution_id=None, agent_run_id="run")
    response = ModelResponse(parts=[ThinkingPart("actual reasoning"), TextPart("actual answer")])

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return response

    class FailAfterResponse(AbstractCapability[None]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position="innermost", wrapped_by=(_AgentRunPersistenceCapability,))

        async def after_node_run(
            self, ctx: RunContext[None], *, node: AgentNode[None], result: NodeResult[None],
        ) -> NodeResult[None]:
            if isinstance(node, ModelRequestNode) and fail_after_node:
                raise RuntimeError("after-node failure")
            return result

    observation = ModelObservationCapability(
        None, source_namespace="history", tenant_id="tenant", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="default", interaction_recorder=recorder,
    )
    persistence = _AgentRunPersistenceCapability(
        recorder=recorder, agent_id="default", agent_run_id="run",
    )
    agent = Agent(FunctionModel(model), capabilities=[observation, persistence, FailAfterResponse()])
    if fail_after_node:
        with pytest.raises(RuntimeError, match="after-node failure"):
            await agent.run("question")
    else:
        assert (await agent.run("question")).output == "actual answer"
    raw_responses = [message for message in recorder.transcript_messages() if isinstance(message, ModelResponse)]
    assert len(raw_responses) == 1
    assert raw_responses[0].parts == response.parts
    assert raw_responses[0].usage == response.usage
    checkpoint = await recorder.latest_checkpoint(include_interrupted=True)
    assert checkpoint is not None
    assert [message for message in checkpoint.messages if isinstance(message, ModelResponse)] == raw_responses
    assert checkpoint.state == ("interrupted" if fail_after_node else "complete")
    completed = [event for event in await store.list_events(agent_run_id="run")
                 if event.event_type == "MODEL_REQUEST_SUCCEEDED"]
    assert len(completed) == 1
    assert completed[0].metadata["linktools.ai.message_seq"] == "2"


@pytest.mark.asyncio
async def test_same_run_recovery_restores_committed_response_ahead_of_checkpoint() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart("question")]))
        await recorder.save_checkpoint(AgentRunCheckpoint(
            recorder.agent_run_id, 1, list(recorder.transcript_messages()),
            transcript_message_count_before=0,
        ))
        fact = journal.begin(1)
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[TextPart("answer")]),
            include_observation=False,
        )
        await history.state.run_store.flush_execution_projection(
            recorder.agent_run_id, execution_id="execution",
        )
        await history.state.run_store.release_staging_many(
            candidate_agent_run_ids=(recorder.agent_run_id,), execution_id="execution",
        )
        recovered = AgentRunRecorder(
            history.state.run_store, execution_id="execution", agent_run_id=recorder.agent_run_id,
            history_boundary=partial(history.state.run_store.flush_execution_projection,
                                     recorder.agent_run_id, execution_id="execution"),
        )
        await recovered.register_agent_run(history.records[0])
        assert len(recovered.transcript_messages()) == 2
        assert recovered.transcript_messages()[1].parts[0].content == "answer"


@pytest.mark.asyncio
async def test_same_run_recovery_rejects_response_fact_without_committed_body() -> None:
    from linktools.ai.errors import AIError, ErrorCode

    async with _history() as history:
        recorder, = history.recorders
        recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart("question")]))
        await recorder.commit_history_boundary()
        await recorder.record_event("MODEL_REQUEST_SUCCEEDED", 1, metadata={
            "linktools.ai.model_request_seq": "1", "linktools.ai.message_seq": "2",
        })
        await recorder.commit_history_boundary()
        await history.state.run_store.release_staging_many(
            candidate_agent_run_ids=(recorder.agent_run_id,), execution_id="execution",
        )
        recovered = AgentRunRecorder(
            history.state.run_store, execution_id="execution", agent_run_id=recorder.agent_run_id,
            history_boundary=partial(history.state.run_store.flush_execution_projection,
                                     recorder.agent_run_id, execution_id="execution"),
        )
        with pytest.raises(AIError) as error:
            await recovered.register_agent_run(history.records[0])
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_trace_keeps_raw_step_event_ordinals_across_filtering_and_archive() -> None:
    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        await recorder.record_event("AGENT_RUN_STARTED", 7)
        fact = journal.begin(7)
        await recorder.record_model_event(fact, phase="started", include_observation=False)
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[TextPart("answer")]),
            include_observation=False,
        )
        await recorder.record_event("AGENT_RUN_SUCCEEDED", 7)

        for archived in (False, True):
            if archived:
                await history.archive()
            first = await history.reader.trace(
                "execution", tenant_id="tenant", cursor=None, limit=1,
                agent_run_seq=1, model_request_seq=1, step_index=7,
            )
            assert first.items[0].step_event_seq == 2
            assert first.items[0].payload["step_index"] == 7
            assert first.next_cursor is not None
            second = await history.reader.trace(
                "execution", tenant_id="tenant", cursor=first.next_cursor, limit=1,
                agent_run_seq=1, model_request_seq=1, step_index=7,
            )
            assert second.items[0].step_event_seq == 3
            assert second.items[0].payload["message_seq"] == 1
            assert second.next_cursor is None
