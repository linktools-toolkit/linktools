#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preparing projection first preserves raw facts and has no owner visibility."""

import asyncio
import hashlib
import random
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._message import SummaryPromptPart, encode_model_messages
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.storage import InMemoryObjectStore

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("variant", ("plain", "summary", "tool", "reordered"))
async def test_projection_first_keeps_raw_transcript_and_exact_projected_context(
    tmp_path: Path, variant: str,
) -> None:
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "state.db"))
    await state.initialize(namespace="projection-first", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    now = run.started_at
    first = ModelRequest(parts=[UserPromptPart("question", timestamp=now)])
    messages = [first, ModelResponse(parts=[TextPart("answer")], timestamp=now)]
    if variant == "tool":
        messages.extend((
            ModelResponse(parts=[ToolCallPart("lookup", {"key": "value"}, "call")], timestamp=now),
            ModelRequest(parts=[ToolReturnPart("lookup", "tool result", "call", timestamp=now)]),
        ))
    context = None
    if variant == "summary":
        context = [ModelRequest(parts=[SummaryPromptPart("trusted summary")]), messages[-1]]
    elif variant == "reordered":
        context = list(reversed(messages))
    try:
        await archive.materialize_checkpoint(run, AgentRunCheckpoint("run", 1, [first], timestamp=now, transcript_message_count_before=0))
        checkpoint = AgentRunCheckpoint("run", 2, messages, timestamp=now, context_messages=context)
        _, batch = await archive.prepare_run_checkpoint(run, checkpoint)
        prepared = batch[0]
        raw_delta = encode_model_messages(messages[1:])
        assert len(prepared.chunks) == 1
        chunk = prepared.chunks[0]
        assert chunk.first_message_index == 1 and chunk.message_count == len(messages) - 1
        assert chunk.raw_digest == hashlib.sha256(raw_delta).hexdigest()
        assert chunk.raw_size == len(raw_delta)
        # Preparing both roots does not advance the owner before its normal CAS.
        head = await archive.transcript_repository.get_head("run")
        assert head.message_count == 1
        await archive.state_store.mutate(lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, prepared))
        raw_messages = [value async for value in archive.transcript_repository.iter_messages("run")]
        assert encode_model_messages(raw_messages) == encode_model_messages(messages)
        loaded = await archive.transcript_repository.load_projected_context("run", prepared.projection)
        assert encode_model_messages(loaded.model_messages()) == encode_model_messages(messages if context is None else context)
    finally:
        await state.close()


async def test_cancel_between_projection_and_chunks_leaves_no_owner_publication_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    objects = InMemoryObjectStore("projection-first")
    state = RuntimeState(RuntimeStorage.sqlite(tmp_path / "state.db", object_store=objects))
    await state.initialize(namespace="projection-first", tenant_id="tenant")
    archive = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
    run = AgentRunRecord("run")
    body = "".join(random.Random(43).choices("abcdefghijklmnopqrstuvwxyz0123456789", k=32768))
    message = ModelRequest(parts=[UserPromptPart(body, timestamp=run.started_at)])
    checkpoint = AgentRunCheckpoint("run", 1, [message], timestamp=run.started_at, transcript_message_count_before=0)
    try:
        async def cancelled(*args, **kwargs):
            raise asyncio.CancelledError

        with monkeypatch.context() as patch:
            patch.setattr(archive, "_prepare_captured_chunks", cancelled)
            with pytest.raises(asyncio.CancelledError):
                await archive.prepare_run_checkpoint(run, checkpoint)
        assert await archive.get_agent_run(agent_run_id="run") is None
        assert await archive.transcript_repository.get_head("run") is None
        assert len([value async for value in objects.list_objects()]) == 1
        # This is the existing content-addressed prepare-time orphan class; retry
        # uses the same object identity and still publishes through the owner.
        await archive.materialize_checkpoint(run, checkpoint)
        assert len([value async for value in objects.list_objects()]) == 1
        assert [value async for value in archive.transcript_repository.iter_messages("run")] == [message]
    finally:
        await state.close()
