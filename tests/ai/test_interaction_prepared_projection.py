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
from linktools.ai.persistence._checkpoint_persistence import CheckpointPersistence
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.persistence._step_archive import StagingAgentRunStore
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord

from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("variant", ("equal", "same_length", "shorter", "reordered", "summary", "tool"))
async def test_prepared_projection_reuse_keeps_exact_request_and_response(
    tmp_path: Path, variant: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="prepared-projection", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
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
        _, batch = await archive.prepare_run_checkpoint(run, AgentRunCheckpoint(
            "run", 1, list(source), timestamp=now, transcript_message_count_before=0,
        ))
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
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(actual,), archive=bundle)
        request, response, request_bytes, _, attempts = await archive.resolve_model_interaction(actual)
        assert attempts is None
        assert request == messages
        assert response == messages[-1:]
        assert request_bytes == envelope
    finally:
        await state.close()


async def test_checkpoint_preparation_preserves_custom_interaction_signature(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="custom-interaction", tenant_id="tenant")
    original = state.run_store.read_store(PersistenceDomain.CHECKPOINT)
    calls = []

    class CustomArchive(ModelInteractionStateStepArchive):
        async def prepare_interactions(
            self, run, interactions, payload, *, local_message_base=0,
            local_message_count=0, local_messages=(), prepared, replay=False,
        ):
            calls.append(tuple(local_messages))
            return await super().prepare_interactions(
                run, interactions, payload, local_message_base=local_message_base,
                local_message_count=local_message_count, local_messages=local_messages,
                prepared=prepared, replay=replay,
            )

    custom = CustomArchive(
        original.state_store, object_store=None, namespace="custom-interaction",
        tenant_id="tenant", runtime_domain=PersistenceDomain.CHECKPOINT,
    )
    try:
        run = AgentRunRecord("run")
        messages = [ModelRequest(parts=[UserPromptPart("request", timestamp=run.started_at)])]
        checkpoint = AgentRunCheckpoint("run", 1, messages, timestamp=run.started_at, transcript_message_count_before=0)
        persistence = CheckpointPersistence(StagingAgentRunStore(), custom, None)
        await persistence.prepare_checkpoint(custom, run, checkpoint, captured_interactions=())
        assert calls == [tuple(messages)]
    finally:
        await state.close()
