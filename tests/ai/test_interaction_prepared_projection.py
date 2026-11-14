#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared projection reuse preserves exact interaction content and custom hooks."""

import hashlib
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.core import UsageMetrics
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._model_interaction import (
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
)
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._transcript_staging import StagedTranscript
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_incremental_projection_boundaries import _storage
from .test_observation_coalescing import _setup

from .test_checkpoint_metadata_snapshot_io import _trace_archive

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("variant", ("equal", "same_length", "shorter", "reordered", "summary", "tool"))
async def test_prepared_projection_reuse_keeps_exact_request_and_response(
    tmp_path: Path, variant: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="prepared-projection", tenant_id="tenant")
    archive = await _trace_archive(state)
    run = AgentRunRecord("run")
    now = run.started_at
    source = (
        ModelRequest(parts=[UserPromptPart("first", timestamp=now)]),
        ModelResponse(parts=[TextPart("reply")], timestamp=now),
        ModelRequest(parts=[UserPromptPart("later", timestamp=now)]),
    )
    messages = source
    if variant == "same_length":
        messages = (ModelRequest(parts=[UserPromptPart("other", timestamp=now)]), *source[1:])
    elif variant == "shorter":
        messages = source[:1]
    elif variant == "reordered":
        messages = source[2:] + source[:2]
    elif variant == "summary":
        messages = (ModelRequest(parts=[UserPromptPart("summary", timestamp=now)]),)
    elif variant == "tool":
        messages = (
            ModelResponse(parts=[ToolCallPart("lookup", {"value": 1}, "call")], timestamp=now),
            ModelRequest(parts=[ToolReturnPart("lookup", "result", "call", timestamp=now)]),
        )
    envelope = b"{}"
    staged = StagedModelInteraction(
        "run", 1, 1, "agent", None, {"route_id": "default"},
        StagedContextProjection((StagedContextSpan(0, len(messages)),)), hashlib.sha256(envelope).hexdigest(),
        StagedContextProjection((StagedContextSpan(len(messages) - 1, len(messages)),)),
        "SUCCEEDED", None, 1, UsageMetrics(model_requests=1), now, now,
    )
    try:
        batch = await archive.prepare_checkpoints(run, (AgentRunCheckpoint(
            "run", 1, list(source), timestamp=now, transcript_message_count_before=0,
        ),))
        expected, = await archive.prepare_interactions(
            run, (staged,), lambda _: envelope, local_message_count=len(messages),
            local_messages=messages, prepared=PreparedArchive(),
        )
        bundle = batch[0].archive
        actual, = await archive.prepare_interactions(
            run, (staged,), lambda _: envelope, local_message_count=len(messages),
            local_messages=messages, prepared=bundle, baseline=batch[0].projection,
        )
        assert actual.request_context.digest == expected.request_context.digest
        assert actual.response_context.digest == expected.response_context.digest
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(actual,), archive=bundle, execution_id="execution")
        request, response, request_bytes, _, attempts = await archive.resolve_model_interaction(actual)
        assert attempts is None
        assert request == messages
        assert response == messages[-1:]
        assert request_bytes == envelope
    finally:
        await state.close()


async def test_trace_publication_calls_current_interaction_override(tmp_path: Path) -> None:
    state = await _setup(tmp_path, "memory")
    original = await _trace_archive(state)
    calls = []

    class CustomArchive(ModelInteractionStateStepArchive):
        async def prepare_interactions(
            self, run, interactions, payload, *, local_message_base=0,
            local_message_count=0, local_messages=(), prepared, replay=False, baseline=None,
        ):
            calls.append(tuple(local_messages))
            return await super().prepare_interactions(
                run, interactions, payload, local_message_base=local_message_base,
                local_message_count=local_message_count, local_messages=local_messages,
                prepared=prepared, replay=replay, baseline=baseline,
            )

    custom = CustomArchive(
        original.state_store, object_store=None, namespace="observation-batches",
        tenant_id="tenant", runtime_domain=PersistenceDomain.TRACE,
        execution_repository=state.execution.executions,
    )
    try:
        state.run_store._archives[PersistenceDomain.TRACE] = custom
        run = await state.run_store.get_agent_run(agent_run_id="run")
        messages = [ModelRequest(parts=[UserPromptPart("request", timestamp=run.started_at)])]
        checkpoint = AgentRunCheckpoint("run", 1, messages, timestamp=run.started_at, transcript_message_count_before=0)
        state.run_store.stage_transcript("run", StagedTranscript(tuple(messages)))
        await state.run_store.save_checkpoint(checkpoint)
        await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert calls == [tuple(messages)]
    finally:
        await state.close()


@pytest.mark.parametrize("variant", ("running", "response", "request", "mixed"))
async def test_interaction_baseline_is_loaded_only_for_request_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str,
) -> None:
    from dataclasses import replace
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="prepared-projection", tenant_id="tenant")
    archive = await _trace_archive(state)
    run = AgentRunRecord("run")
    messages = (ModelResponse(parts=[TextPart("reply")], timestamp=run.started_at),)
    context = StagedContextProjection((StagedContextSpan(0, 1),))
    staged = StagedModelInteraction(
        "run", 1, 1, "agent", None, {},
        context if variant in {"running", "request", "mixed"} else None,
        hashlib.sha256(b"{}").hexdigest() if variant in {"running", "request", "mixed"} else None,
        None if variant == "running" else context, "RUNNING" if variant == "running" else "SUCCEEDED",
        None, None if variant == "running" else 1,
        None if variant == "running" else UsageMetrics(model_requests=1),
        run.started_at, None if variant == "running" else run.started_at,
    )
    interactions = (replace(staged, request_context=None, request_envelope_digest=None, model_request_seq=2), staged) if variant == "mixed" else (staged,)
    calls = []
    load = archive._history.load_projection
    async def tracked(*args, **kwargs):
        calls.append(args)
        return await load(*args, **kwargs)
    monkeypatch.setattr(archive._history, "load_projection", tracked)
    try:
        result = await archive.prepare_interactions(
            run, interactions, lambda _: b"{}", local_message_count=1,
            local_messages=messages, prepared=PreparedArchive(),
        )
        assert len(result) == len(interactions)
        assert len(calls) == (1 if variant in {"request", "mixed"} else 0)
        assert all((value.response_context is not None) == (variant != "running") for value in result)
        assert (result[-1].request_context is not None) == (variant in {"request", "mixed"})
    finally:
        await state.close()
