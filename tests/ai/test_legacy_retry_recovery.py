#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recorded retry bytes and pending identities survive native replay."""

import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import decode_model_messages, encode_model_messages
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import _history as transcript
from linktools.ai.persistence.runs import StagedTranscript
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder

from .test_history_request_association import _history

pytestmark = [pytest.mark.asyncio, pytest.mark.filterwarnings("error:`RetryPromptPart` is deprecated")]
_FIXTURE = Path(__file__).parent / "fixtures" / "persistence" / "legacy_retry_pending_v1.json"


async def test_legacy_pending_retry_retains_bytes_and_cursor_through_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _FIXTURE.read_bytes()
    provenance = json.loads(_FIXTURE.with_suffix(".provenance.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance["sha256"]
    message, = decode_model_messages(raw)
    retry, = message.parts
    assert isinstance(retry, ToolReturnPart) and retry.outcome == "retried"
    assert retry.tool_name == "echo" and retry.tool_call_id == "call"

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
            recorder.transcript_messages(), ModelRequest(parts=[retry]), ("retry:call",),
        ))

        def recorded_writer(messages):
            # Replay real writer bytes only for the frozen pending part.
            if len(messages) == 1 and list(messages[0].parts) == [retry]:
                return raw
            return encode_model_messages(messages)

        with monkeypatch.context() as patch:
            patch.setattr(transcript, "encode_model_messages", recorded_writer)
            await recorder.commit_history_boundary()
        archive_store = history.state.run_store.read_store(PersistenceDomain.TRACE)
        repository = archive_store.transcript_repository
        head = await repository.get_head(recorder.agent_run_id)
        chunks, keys = await archive_store.state_store.read(
            lambda tx: repository.capture_pending_in_transaction(tx, head),
        )
        assert keys == ("retry:call",)
        content_archive = await repository.content_archive()
        original = await content_archive.read_ranges((chunks[0].content,), codec="model-message")
        assert original == ((raw,),)
        with pytest.raises(AIError) as corrupt:
            await repository._decode_chunk_messages(replace(chunks[0], raw_digest="0" * 64))
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
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
        assert await content_archive.read_ranges((chunks[0].content,), codec="model-message") == original
        assert _FIXTURE.read_bytes() == raw
        results = [part for item in recorder.transcript_messages() for part in item.parts
                   if isinstance(part, ToolReturnPart)]
        assert results == [retry]
        assert b'"retry-prompt"' not in encode_model_messages(recorder.transcript_messages())
