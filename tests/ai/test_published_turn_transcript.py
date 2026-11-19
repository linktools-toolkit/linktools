#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A published turn can reuse only its exact local transcript delta."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import (
    ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart,
)

from linktools.ai.persistence import MessageArchive
from linktools.ai.persistence._archive import archive_content_scope
from linktools.ai.persistence.contracts import (
    ContextProjection, StoredAgentRunCheckpoint, TranscriptChunk, TranscriptOrigin,
)
from linktools.ai.persistence.runs import ArchiveRoot, PreparedAgentRunCheckpoint, PreparedAgentRunCheckpointBatch
from linktools.ai.session import SessionTurnCommit


@pytest.fixture
def prepared_turn() -> tuple[PreparedAgentRunCheckpointBatch, SessionTurnCommit]:
    archive = MessageArchive()
    messages = (
        ModelResponse(parts=[ThinkingPart("reasoning"), ToolCallPart("lookup", {"key": "x"}, "call")]),
        ModelRequest(parts=[ToolReturnPart("lookup", "found", "call")]),
        ModelResponse(parts=[TextPart("answer")]),
    )
    delta = archive.snapshot(messages)
    root = ArchiveRoot("group", archive_content_scope("namespace", "tenant"))
    offsets = ((6, delta[:2]), (8, delta[2:]))
    chunks = tuple(
        TranscriptChunk("history", first, len(snapshot), TranscriptOrigin.RAW, "raw",
                        *snapshot.encoded_digest(), root)
        for first, snapshot in offsets
    )
    projection = ContextProjection(())
    checkpoint = PreparedAgentRunCheckpoint(
        "history", StoredAgentRunCheckpoint("run", 1, datetime.now(timezone.utc), "complete", projection, 9),
        chunks, projection, context_messages=archive.concatenate(
            archive.snapshot([ModelResponse(parts=[TextPart("previous content")])]), delta,
        ), transcript_delta=delta,
    )
    return PreparedAgentRunCheckpointBatch("run", (checkpoint,), 0, 0, 9), SessionTurnCommit(
        "session", 3, "execution", 11, 14,
    )


def test_local_delta_is_exact_and_excludes_inherited_context(prepared_turn) -> None:
    batch, commit = prepared_turn
    messages = batch.session_turn_messages(commit, inherited_message_count=5)
    assert messages is not None
    assert [part.content for message in messages for part in message.parts if isinstance(
        part, (ThinkingPart, ToolReturnPart, TextPart))] == ["reasoning", "found", "answer"]
    assert isinstance(messages[0].parts[1], ToolCallPart)
    assert messages[0].parts[1].tool_call_id == messages[1].parts[0].tool_call_id == "call"
    assert batch.session_turn_messages(replace(commit, start_message_index=12), inherited_message_count=5) is None
    assert batch.session_turn_messages(commit, inherited_message_count=4) is None


def test_delta_requires_complete_local_coverage(prepared_turn) -> None:
    batch, commit = prepared_turn
    checkpoint = batch.checkpoints[0]
    assert checkpoint.transcript_delta is not None
    for candidate in (
        replace(checkpoint, transcript_delta=None),
        replace(checkpoint, transcript_delta=checkpoint.transcript_delta[1:]),
        replace(checkpoint, chunks=checkpoint.chunks[:1]),
        replace(checkpoint, chunks=(checkpoint.chunks[0], replace(checkpoint.chunks[1], first_message_index=9))),
    ):
        assert replace(batch, checkpoints=(candidate,)).session_turn_messages(
            commit, inherited_message_count=5,
        ) is None


def test_modified_projected_context_cannot_replace_local_transcript(prepared_turn) -> None:
    batch, commit = prepared_turn
    checkpoint = batch.checkpoints[0]
    archive = MessageArchive()
    changed = replace(checkpoint, context_messages=archive.snapshot([
        ModelResponse(parts=[TextPart("trimmed and replaced context")]),
    ]))
    messages = replace(batch, checkpoints=(changed,)).session_turn_messages(commit, inherited_message_count=5)
    assert messages is not None and checkpoint.transcript_delta is not None
    assert messages.encoded_digest() == checkpoint.transcript_delta.encoded_digest()
