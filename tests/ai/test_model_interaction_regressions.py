#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model interaction paging, public projection, and replay invariants."""

import hashlib
from dataclasses import replace

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    FilePart,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._model_interaction import (
    StagedContextInline,
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    build_context_projection,
    model_response_projection,
    project_public_messages,
)
from linktools.ai.runtime.state import RuntimeDomain, RuntimeRetentionMode
from linktools.ai.runtime.state._model_interaction_runtime import (
    ModelInteractionRuntimeStepStore,
)
from linktools.ai.runtime.state._model_interaction_store import (
    ModelInteractionInMemoryStepArchive,
    ModelInteractionStagingStepStore,
)
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.runtime.state._steps import (
    InMemoryStepArchive,
    RuntimeStepStore,
    StagingStepStore,
    _ProjectionOffset,
)


def _interaction(sequence: int) -> StagedModelInteraction:
    return StagedModelInteraction(
        run_id="run",
        step_index=1,
        request_sequence=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=StagedContextProjection(0, "0" * 64, ()),
        request_envelope_digest="a" * 64,
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=0,
        usage=None,
    )


def test_public_projection_preserves_business_binary_mapping() -> None:
    content = {"nested": [{"kind": "binary", "data": "keep"}]}
    message = ModelRequest(
        parts=[ToolReturnPart("tool", content, tool_call_id="call-1")]
    )
    projected = project_public_messages((message,))
    assert projected[0]["parts"][0]["content"] == content
    assert content == {"nested": [{"kind": "binary", "data": "keep"}]}


@pytest.mark.parametrize("response", (False, True))
def test_public_projection_summarizes_real_binary_without_mutation(
    response: bool,
) -> None:
    raw = b"\x00\xff\xfbimage"
    binary = BinaryContent(data=raw, media_type="image/png")
    if response:
        projected = model_response_projection(
            ModelResponse(parts=[FilePart(content=binary)])
        )
        content = projected["parts"][0]["content"]
    else:
        message = ModelRequest(parts=[UserPromptPart(content=["image", binary])])
        projected = project_public_messages((message,))[0]
        content = projected["parts"][0]["content"][1]
    assert "data" not in content
    assert content["size"] == len(raw)
    assert content["digest"] == hashlib.sha256(raw).hexdigest()
    assert binary.data == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("enhanced", (False, True))
async def test_base_step_store_applies_interaction_page_contract(
    enhanced: bool,
) -> None:
    store = ModelInteractionStagingStepStore() if enhanced else StagingStepStore()
    await store.initialize()
    try:
        await store.register_run(RunRecord("run"))
        interactions = tuple(_interaction(sequence) for sequence in range(1, 4))
        for interaction in interactions:
            store.stage_model_interaction(interaction)
        assert await store.list_model_interactions(
            run_id="run", after_request_sequence=1, limit=1,
        ) == [interactions[1]]
        assert await store.list_model_interactions(
            run_id="run", after_request_sequence=3, limit=1,
        ) == []
        assert await store.list_model_interactions(run_id="run") == list(interactions)
        with pytest.raises(ValueError):
            await store.list_model_interactions(run_id="run", limit=0)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enhanced", (False, True))
async def test_runtime_step_store_pages_plain_staging(enhanced: bool) -> None:
    store_type = ModelInteractionRuntimeStepStore if enhanced else RuntimeStepStore
    store = store_type(
        StagingStepStore(),
        conversation_archive=InMemoryStepArchive(RuntimeDomain.CONVERSATION),
        execution_archive=None,
        recovery_archive=None,
        conversation_retention=RuntimeRetentionMode.VOLATILE,
        execution_retention=RuntimeRetentionMode.VOLATILE,
        recovery_retention=RuntimeRetentionMode.VOLATILE,
    )
    await store.initialize()
    try:
        await store.register_run(RunRecord("run"))
        values = tuple(_interaction(sequence) for sequence in range(1, 4))
        for value in values:
            store.stage_model_interaction(value)
        assert await store.list_model_interactions(
            run_id="run", after_request_sequence=1, limit=1,
        ) == [values[1]]
    finally:
        await store.preflight_close()
        await store.close()


def test_repeated_message_matching_preserves_first_unused_source() -> None:
    message = ModelRequest(parts=[UserPromptPart(content="same")])
    other = ModelRequest(parts=[UserPromptPart(content="other")])
    store = StagingStepStore()
    projection = build_context_projection(
        (message, message, other),
        (message, other, message, message),
        lambda content: store.intern_payload("run", content),
    )
    assert projection.items[:3] == (
        StagedContextSpan(0, 1),
        StagedContextSpan(2, 3),
        StagedContextSpan(1, 2),
    )
    assert len(projection.items) == 4
    assert isinstance(projection.items[3], StagedContextInline)


@pytest.mark.asyncio
async def test_interaction_replay_preserves_nonzero_sequence_origin() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    try:
        values = tuple(_interaction(sequence) for sequence in (5, 6, 7))
        for value in values:
            store.stage_model_interaction(value)
        store.stage_model_interaction(values[1])
        for invalid in (
            replace(values[1], duration_ns=1),
            _interaction(9),
            _interaction(4),
        ):
            with pytest.raises(AIError) as raised:
                store.stage_model_interaction(invalid)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await store.list_model_interactions(run_id="run") == list(values)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_interaction_capture_respects_durable_high_water() -> None:
    store = ModelInteractionStagingStepStore()
    await store.initialize()
    try:
        await store.register_run(RunRecord("run"))
        values = tuple(_interaction(sequence) for sequence in (5, 6, 7))
        for value in values:
            store.stage_model_interaction(value)

        captured = store.capture_projection_local(
            "run",
            _ProjectionOffset(interactions=5),
        )

        assert captured is not None
        assert captured.interactions == values[1:]
        assert captured.base_interaction_offset == 5
        assert captured.target_interaction_offset == 7
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_volatile_archive_does_not_expose_staged_interactions() -> None:
    archive = ModelInteractionInMemoryStepArchive(RuntimeDomain.EXECUTION)
    await archive.initialize()
    try:
        archive.stage_model_interaction(_interaction(1))
        with pytest.raises(AIError) as raised:
            await archive.list_model_interactions(run_id="run")
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await archive.close()
