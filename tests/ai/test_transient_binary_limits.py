#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate transient BinaryContent request limits."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import AgentContext
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._compaction import RuntimeCompaction
from linktools.ai.workspace import Workspace, WorkspacePolicy


@pytest.mark.asyncio
async def test_pending_binary_limit_applies_to_combined_model_context() -> None:
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
    context = RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )
    request_context = ModelRequestContext(
        model=TestModel(),
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

    with pytest.raises(AIError) as raised:
        await RuntimeCompaction(None).before_model_request(context, request_context)

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
