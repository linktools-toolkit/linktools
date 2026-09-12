#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient BinaryContent projection contracts."""

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from linktools.ai.runtime._harness import HarnessStepStoreAdapter
from linktools.ai.runtime._message import (
    binary_content_usage,
    project_transient_binary_content,
)
from linktools.ai.runtime.state._step_contracts import (
    ContinuableSnapshot,
    RunRecord,
)
from linktools.ai.runtime.state._steps import StagingStepStore


def _request(name: str, body: bytes) -> ModelRequest:
    return ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    f"Workspace file: {name}",
                    BinaryContent(body, media_type="image/png", identifier=name),
                ]
            )
        ]
    )


def test_pending_binary_is_kept_until_complete_model_response() -> None:
    request = _request("a.png", b"image")
    interrupted = ModelResponse(parts=[TextPart("partial")], state="interrupted")

    pending = project_transient_binary_content((request,))
    interrupted_projection = project_transient_binary_content((request, interrupted))

    assert pending == (request,)
    assert interrupted_projection == (request, interrupted)
    assert binary_content_usage(pending) == (1, 5)


def test_consumed_binary_is_removed_without_mutating_raw_transcript() -> None:
    request = _request("a.png", b"image")
    response = ModelResponse(parts=[TextPart("done")])
    raw: tuple[ModelMessage, ...] = (request, response)

    projected = project_transient_binary_content(raw)

    assert projected != raw
    assert binary_content_usage(projected) == (0, 0)
    assert binary_content_usage(raw) == (1, 5)
    projected_request = projected[0]
    assert isinstance(projected_request, ModelRequest)
    content = projected_request.parts[0].content  # type: ignore[attr-defined]
    assert "Workspace file: a.png" in content
    assert any(
        isinstance(item, str) and "binary content already consumed" in item
        for item in content
    )


def test_only_binary_after_latest_complete_response_remains_pending() -> None:
    first = _request("old.png", b"old")
    response = ModelResponse(parts=[TextPart("done")])
    second = _request("new.png", b"new")

    projected = project_transient_binary_content((first, response, second))

    assert binary_content_usage(projected) == (1, 3)
    assert binary_content_usage((first, response, second)) == (2, 6)


def test_snapshot_context_projects_consumed_binary_even_without_compaction() -> None:
    adapter = HarnessStepStoreAdapter(object(), execution_id="execution")  # type: ignore[arg-type]
    request = _request("a.png", b"image")
    response = ModelResponse(parts=[TextPart("done")])
    raw = [request, response]

    context = adapter.snapshot_context_messages(raw)

    assert context is not None
    assert binary_content_usage(context) == (0, 0)
    assert binary_content_usage(raw) == (1, 5)


def test_snapshot_context_composes_compaction_with_pending_binary() -> None:
    adapter = HarnessStepStoreAdapter(object(), execution_id="execution")  # type: ignore[arg-type]
    consumed_request = _request("old.png", b"old")
    consumed_response = ModelResponse(parts=[TextPart("done")])
    source = (consumed_request, consumed_response)
    summary = ModelRequest(parts=[UserPromptPart(content="summary")])
    adapter.remember_context_projection(source, (summary,))
    pending = _request("new.png", b"new")

    context = adapter.snapshot_context_messages((*source, pending))

    assert context is not None
    assert context[0] == summary
    assert context[-1] == pending
    assert binary_content_usage(context) == (1, 3)


@pytest.mark.asyncio
async def test_snapshot_recovery_keeps_pending_and_drops_consumed_binary() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("pending"))
    await store.register_run(RunRecord("consumed"))
    adapter = HarnessStepStoreAdapter(store, execution_id=None)
    pending_request = _request("pending.png", b"pending")
    consumed_request = _request("consumed.png", b"consumed")
    consumed_response = ModelResponse(parts=[TextPart("done")])

    await store.save_snapshot(
        ContinuableSnapshot(
            run_id="pending",
            step_index=1,
            messages=[pending_request],
            context_messages=adapter.snapshot_context_messages([pending_request]),
        )
    )
    consumed_raw = [consumed_request, consumed_response]
    await store.save_snapshot(
        ContinuableSnapshot(
            run_id="consumed",
            step_index=1,
            messages=consumed_raw,
            context_messages=adapter.snapshot_context_messages(consumed_raw),
        )
    )

    pending = await store.load_loaded_model_context(owner_id="pending")
    consumed = await store.load_loaded_model_context(owner_id="consumed")

    assert binary_content_usage(pending.model_messages()) == (1, 7)
    assert binary_content_usage(consumed.model_messages()) == (0, 0)
