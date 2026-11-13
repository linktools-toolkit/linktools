#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recorded retry bytes and pending identities survive native replay."""

from dataclasses import replace
from functools import partial

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.runs import StagedTranscript
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder

from .test_history_request_association import _history

pytestmark = pytest.mark.asyncio


async def test_pending_retry_retains_bytes_and_cursor_through_recovery() -> None:
    retry = ToolReturnPart("echo", "retry result", tool_call_id="call", outcome="retried")

    async with _history() as history:
        recorder, = history.recorders
        journal, = history.journals
        recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart("first"), UserPromptPart("second")]))
        fact = journal.begin(1)
        call = ToolCallPart("echo", {}, tool_call_id="call")
        await recorder.record_model_event(
            journal.finish(fact.model_request_seq, status="SUCCEEDED"),
            phase="completed", response=ModelResponse(parts=[call]), include_observation=False,
        )
        await recorder.record_tool_start(call, 1)
        history.state.run_store.stage_transcript(recorder.agent_run_id, StagedTranscript(
            recorder.transcript_messages(), ModelRequest(parts=[retry]), ("tool_result:call",),
        ))

        await recorder.commit_history_boundary()
        archive_store = history.state.run_store.read_store(PersistenceDomain.TRACE)
        repository = archive_store.transcript_repository
        head = await repository.get_head(recorder.agent_run_id)
        chunks, keys = await archive_store.state_store.read(
            lambda tx: repository.capture_pending_in_transaction(tx, head),
        )
        assert keys == ("tool_result:call",)
        original = await repository._read_message_roots((chunks[0].content,))
        assert await repository._decode_chunk_messages(replace(chunks[0], raw_digest="0" * 64)) == await repository._decode_chunk_messages(chunks[0])
        first = await history.reader.history("execution", tenant_id="tenant", cursor=None, limit=3)
        assert first.next_cursor is not None
        first_transcript = await history.reader.transcript("execution", tenant_id="tenant", cursor=None, limit=1)
        assert first_transcript.next_cursor is not None

        recorder = AgentRunRecorder(
            history.state.run_store, execution_id="execution", agent_run_id=recorder.agent_run_id,
            history_boundary=partial(history.state.run_store.flush_execution_projection,
                                     recorder.agent_run_id, execution_id="execution"),
        )
        await recorder.register_agent_run(history.records[0])
        await recorder.record_tool_result_boundary(retry, 2)
        recorder.append_transcript_message(ModelRequest(parts=[retry, UserPromptPart("new input")], instructions="next instructions"))
        await recorder.commit_history_boundary()
        history.recorders = (recorder,)
        await history.archive()
        page = await history.reader.history(
            "execution", tenant_id="tenant", cursor=first.next_cursor, limit=10,
        )
        assert [(item.item_kind, item.tool_call_id, item.content, item.status) for item in page.items] == [
            ("tool_result", "call", "retry result", "STARTED"),
        ]
        assert page.items[0].model_request_seq == 1 and page.items[0].step_index == 1
        assert page.next_cursor is None
        continuation = await history.reader.transcript(
            "execution", tenant_id="tenant", cursor=first_transcript.next_cursor, limit=10,
        )
        assert [item.text for item in continuation.items] == ["second"]
        assert continuation.next_cursor is None
        latest = await history.reader.history(
            "execution", tenant_id="tenant", cursor=None, limit=10, tool_call_id="call",
        )
        assert [item.status for item in latest.items] == ["FAILED", "FAILED"]
        assert await repository._read_message_roots((chunks[0].content,)) == original
        results = [part for item in recorder.transcript_messages() for part in item.parts
                   if isinstance(part, ToolReturnPart)]
        assert results == [retry]
