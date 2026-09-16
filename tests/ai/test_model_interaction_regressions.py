#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for model interaction paging and public projection."""

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from linktools.ai.runtime._model_interaction import (
    StagedContextProjection,
    StagedModelInteraction,
    project_public_messages,
)
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.runtime.state._steps import StagingStepStore


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
    message = ModelRequest(
        parts=[
            ToolReturnPart(
                "tool",
                {"kind": "binary", "data": "keep"},
                tool_call_id="call-1",
            )
        ]
    )

    projected = project_public_messages((message,))
    content = projected[0]["parts"][0]["content"]  # type: ignore[index]
    assert content == {"kind": "binary", "data": "keep"}


@pytest.mark.asyncio
async def test_base_step_store_applies_interaction_page_contract() -> None:
    store = StagingStepStore()
    await store.initialize()
    await store.register_run(RunRecord("run"))
    interactions = tuple(_interaction(sequence) for sequence in range(1, 4))
    for interaction in interactions:
        store.stage_model_interaction(interaction)

    assert await store.list_model_interactions(
        run_id="run",
        after_request_sequence=1,
        limit=1,
    ) == [interactions[1]]
