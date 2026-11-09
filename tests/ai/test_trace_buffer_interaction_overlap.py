#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlapping Trace captures retain each completed request's immutable roots."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart,
)

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._model_interaction import (
    StagedContextProjection, StagedContextSpan, StagedModelInteraction,
)
from linktools.ai.persistence._projection_payload import decode_projection_payload
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.persistence.runs import RuntimeAgentRunStore

from .test_observation_coalescing import _setup


def _messages() -> tuple[ModelMessage, ...]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        ModelRequest(parts=[UserPromptPart("question", timestamp=now)]),
        ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="first")], timestamp=now),
        ModelRequest(parts=[ToolReturnPart("work", "first result", tool_call_id="first", timestamp=now)]),
        ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id="second")], timestamp=now),
        ModelRequest(parts=[ToolReturnPart("work", "second result", tool_call_id="second", timestamp=now)]),
        ModelResponse(parts=[TextPart("answer")], timestamp=now),
    )


def _interaction(store: RuntimeAgentRunStore, sequence: int, *, running: bool = False) -> StagedModelInteraction:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    envelope, _ = store.intern_payload("run", b"{}")
    return StagedModelInteraction(
        "run", sequence, sequence, "agent", None, {"model": "test"},
        StagedContextProjection((StagedContextSpan(0, sequence * 2 - 1),)), envelope,
        None if running else StagedContextProjection((StagedContextSpan(sequence * 2 - 1, sequence * 2),)),
        "RUNNING" if running else "SUCCEEDED", None, None if running else 1,
        None, now, None if running else now,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "local_sqlite"))
@pytest.mark.parametrize("overlap,progress", (
    (False, "unchanged"), (True, "unchanged"), (True, "running"), (True, "new"),
))
async def test_trace_buffer_overlap_preserves_completed_requests_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, overlap: bool, progress: str,
) -> None:
    state = await _setup(tmp_path, backend)
    store = state.run_store
    owner = store.trace_persistence
    archive = store.read_store(PersistenceDomain.TRACE)
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    messages = _messages()
    entered, release = asyncio.Event(), asyncio.Event()
    prepare = archive.prepare_interactions
    first = None
    preparations = 0

    async def held(*args, **kwargs):
        nonlocal preparations
        preparations += 1
        if preparations == 1:
            entered.set()
            await release.wait()
        return await prepare(*args, **kwargs)

    monkeypatch.setattr(archive, "prepare_interactions", held)
    try:
        store.stage_transcript("run", StagedTranscript(messages[:5 if progress == "running" else 4]))
        for sequence in (1, 2):
            store.stage_model_interaction(_interaction(store, sequence))
        if progress == "running":
            store.stage_model_interaction(_interaction(store, 3, running=True))
        await store.append_event(StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 2))
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        buffer = owner._buffers["run"]
        first = asyncio.create_task(owner._automatic_flush(buffer))
        await asyncio.wait_for(entered.wait(), 5)
        assert "run" in store._durability_flights
        if not overlap:
            release.set()
            await asyncio.wait_for(first, 10)
            assert buffer.error is None
        if progress != "unchanged":
            store.stage_transcript("run", StagedTranscript(messages))
            store.stage_model_interaction(_interaction(store, 3))
        await store.append_event(StepEvent("run", "TOOL_CALL_STARTED", 2, tool_call_id="second", tool_name="work"))
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        queued = await buffer.backend.read_batch(buffer.id, limit=10)
        captures = tuple(decode_projection_payload(entry[2]) for entry in queued)
        if overlap:
            assert len(captures) == 2
            assert captures[0].interactions[:2] == captures[1].interactions[:2]
        else:
            assert captures[-1].interactions == ()
        release.set()
        await asyncio.wait_for(first, 10)
        assert buffer.error is None
        before = await archive.list_model_interactions(agent_run_id="run")
        before_contexts = await archive.resolve_model_interactions(before[:2])
        if overlap:
            captured = captures[-1]
            # Preparing the identical terminal input from the latest request's
            # baseline alone produces a different reference, despite equal bytes.
            regenerated = await prepare(
                captured.run, captured.interactions[:2], dict(captured.payloads).__getitem__,
                local_message_count=len(captured.local_messages), local_messages=captured.local_messages,
                prepared=PreparedArchive(), baseline=before[1].request_context,
            )
            old, new = before[0].request_context.items[0], regenerated[0].request_context.items[0]
            assert old.root != new.root
            assert (old.raw_digest, old.raw_size) == (new.raw_digest, new.raw_size)
        await owner._automatic_flush(buffer)
        assert buffer.error is None
        after = await archive.list_model_interactions(agent_run_id="run")
        assert after[:2] == before[:2]
        assert await archive.resolve_model_interactions(after[:2]) == before_contexts
        assert len(after) == (2 if progress == "unchanged" else 3)
        assert all(record.status == "SUCCEEDED" for record in after)
        if progress == "running":
            assert before[2].status == "RUNNING" and before[2].request_context is None
        if progress != "unchanged":
            request, response, *_ = await archive.resolve_model_interaction(after[2])
            assert tuple(request) == messages[:5]
            assert tuple(response) == messages[5:]
        assert len(await archive.list_events(agent_run_id="run")) == 2
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await archive.list_model_interactions(agent_run_id="run") == after
        assert not store._durability_flights
    finally:
        release.set()
        if first is not None:
            await asyncio.gather(first, return_exceptions=True)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "local_sqlite"))
@pytest.mark.parametrize("change", ("identity", "request", "terminal"))
async def test_trace_buffer_replay_rejects_changed_terminal_facts(
    tmp_path: Path, backend: str, change: str,
) -> None:
    state = await _setup(tmp_path, backend)
    store = state.run_store
    owner = store.trace_persistence
    archive = store.read_store(PersistenceDomain.TRACE)
    owner.configure("run", buffer=owner._caches["run"], batch_size=128, flush_interval=60)
    try:
        store.stage_transcript("run", StagedTranscript(_messages()[:4]))
        for sequence in (1, 2):
            store.stage_model_interaction(_interaction(store, sequence))
        await store.append_event(StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 2))
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1, deferred=True)
        buffer = owner._buffers["run"]
        queued = await buffer.backend.read_batch(buffer.id, limit=10)
        captured = decode_projection_payload(queued[0][2])
        await owner._automatic_flush(buffer)
        assert buffer.error is None
        original = await archive.list_model_interactions(agent_run_id="run")
        if change == "request":
            first, *rest = captured.local_messages
            changed = replace(first, parts=[UserPromptPart("changed", timestamp=first.parts[0].timestamp)])
            captured = replace(captured, local_messages=(changed, *rest))
        else:
            first, *rest = captured.interactions
            changed = replace(first, **({"step_index": 10} if change == "identity" else {"duration_ns": 99}))
            captured = replace(captured, interactions=(changed, *rest))
        with pytest.raises(AIError) as error:
            await store.publish_buffered_execution_projection(captured, execution_id="execution")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert not store._durability_flights
        assert await archive.list_model_interactions(agent_run_id="run") == original
        assert len(await archive.list_events(agent_run_id="run")) == 1
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "local_sqlite"))
@pytest.mark.parametrize("outcome", ("rollback", "unknown", "cancelled"))
async def test_trace_buffer_overlap_keeps_durable_outcome_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, outcome: str,
) -> None:
    state = await _setup(tmp_path, backend)
    group = state.execution.executions.state_store.storage_group
    store = state.run_store
    archive = store.read_store(PersistenceDomain.TRACE)
    entered, release = asyncio.Event(), asyncio.Event()
    caller = None
    unresolved = False
    try:
        store.stage_transcript("run", StagedTranscript(_messages()[:4]))
        for sequence in (1, 2):
            store.stage_model_interaction(_interaction(store, sequence))
        await store.append_event(StepEvent("run", "MODEL_REQUEST_SUCCEEDED", 2))
        first = await store.capture_buffered_execution_projection("run", execution_id="execution", producer_generation=1)
        assert first is not None
        await store.append_event(StepEvent("run", "TOOL_CALL_STARTED", 2, tool_call_id="second", tool_name="work"))
        second = await store.capture_buffered_execution_projection("run", execution_id="execution", producer_generation=1)
        assert second is not None
        first, second = decode_projection_payload(first), decode_projection_payload(second)
        assert first.interactions == second.interactions
        await store.publish_buffered_execution_projection(first, execution_id="execution")
        original = await archive.list_model_interactions(agent_run_id="run")
        commit, head = archive.sync_prepared_projection, archive.execution_history_head_record
        marker = AIError(ErrorCode.STORAGE_UNAVAILABLE, "transaction rolled back")
        submitted = False

        async def interrupted(*args, **kwargs):
            nonlocal submitted
            await commit(*args, **kwargs)
            submitted = True
            if outcome == "cancelled":
                entered.set()
                await release.wait()
            else:
                raise marker

        async def readback(*args, **kwargs):
            if submitted and outcome == "unknown":
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            return await head(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(archive, "sync_prepared_projection", interrupted)
            patch.setattr(archive, "execution_history_head_record", readback)
            if outcome == "cancelled":
                caller = asyncio.create_task(store.publish_buffered_execution_projection(second, execution_id="execution"))
                await asyncio.wait_for(entered.wait(), 5)
                caller.cancel()
                await asyncio.sleep(0)
                caller.cancel()
                await asyncio.sleep(0)
                assert not caller.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            elif outcome == "rollback":
                with pytest.raises(AIError) as error:
                    await store.publish_buffered_execution_projection(second, execution_id="execution")
                assert error.value is marker
            else:
                with pytest.raises(AIError) as error:
                    await store.publish_buffered_execution_projection(second, execution_id="execution")
                assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
                unresolved = True
        assert submitted
        assert await archive.list_model_interactions(agent_run_id="run") == original
        if outcome != "cancelled":
            assert len(await archive.list_events(agent_run_id="run")) == 1
        if unresolved:
            flight = store._durability_flights["run"]
            with pytest.raises(AIError) as fenced:
                await store.wait_projection_flight("run")
            assert fenced.value is error.value
            assert store._durability_flights["run"] is flight
            with pytest.raises(AIError) as closing:
                await state.close()
            assert closing.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            return
        assert not store._durability_flights
        # Known rollback retries and already-finalized captures both use the
        # original bytes while preserving the source-confirmed terminal roots.
        await store.publish_buffered_execution_projection(second, execution_id="execution")
        await store.publish_buffered_execution_projection(second, execution_id="execution")
        assert await archive.list_model_interactions(agent_run_id="run") == original
        assert len(await archive.list_events(agent_run_id="run")) == 2
        assert not store._durability_flights
    finally:
        release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        if unresolved:
            await group.close()
            if backend != "memory":
                await group.context.close()
        else:
            await state.close()
