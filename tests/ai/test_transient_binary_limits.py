#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate transient BinaryContent request limits."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import TieredCompaction

from linktools.ai.capability import AgentContext
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._compaction import RuntimeCompaction
from linktools.ai.workspace import Workspace, WorkspacePolicy


def _contexts() -> tuple[RunContext[AgentContext[None]], ModelRequestContext]:
    workspace = Workspace(
        root=Path("."),
        config={},
        workspace_id="workspace",
        policy=WorkspacePolicy(max_binary_input_parts=1),
    )
    deps = AgentContext(
        app=None,
        principal=Principal("user", "tenant", "local_trusted"),
        workspace=workspace,
        session_id=None,
        execution_id="execution",
        session_metadata={},
    )
    model = TestModel()
    context = RunContext(
        deps=deps,
        model=model,
        usage=RunUsage(),
        run_id="run",
    )
    request_context = ModelRequestContext(
        model=model,
        messages=[
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            BinaryContent(b"a", media_type="image/png"),
                            BinaryContent(b"b", media_type="image/png"),
                        ]
                    )
                ]
            )
        ],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    return context, request_context


async def _unexpected_handler(_request_context: ModelRequestContext) -> ModelResponse:
    raise AssertionError("provider handler must not run for an invalid model request")


@pytest.mark.asyncio
async def test_pending_binary_limit_applies_to_combined_model_context() -> None:
    context, request_context = _contexts()

    with pytest.raises(AIError) as raised:
        await RuntimeCompaction(None).wrap_model_request(
            context,
            request_context=request_context,
            handler=_unexpected_handler,
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_pending_binary_limit_fails_before_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, request_context = _contexts()

    async def unexpected_compaction(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compaction must not run for an invalid model request")

    monkeypatch.setattr(TieredCompaction, "before_model_request", unexpected_compaction)

    with pytest.raises(AIError) as raised:
        await RuntimeCompaction(1).wrap_model_request(
            context,
            request_context=request_context,
            handler=_unexpected_handler,
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
