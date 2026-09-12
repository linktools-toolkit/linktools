#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end transient file behavior across a Pydantic agent loop."""

from pathlib import Path

import pytest
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import AgentContext, workspace_capabilities
from linktools.ai.core import Principal
from linktools.ai.runtime._compaction import RuntimeCompaction
from linktools.ai.runtime._message import binary_content_usage
from linktools.ai.workspace import Workspace


class _Session:
    async def canonicalize_path(self, path: str) -> str:
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        assert path == "evidence.png"
        value = b"png"
        assert max_bytes is None or len(value) <= max_bytes
        return value

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        assert path == "note.txt"
        assert offset == 0
        assert limit is None
        return "note"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_attach_files_is_visible_for_one_model_request_only(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _Session()
    toolset = workspace_capabilities(
        workspace,
        ("attach_files", "read_file"),
        session=session,  # type: ignore[arg-type]
    )[0].get_toolset()
    observed_binary: list[tuple[int, int]] = []

    def model_function(
        messages: list[ModelMessage],
        _info: AgentInfo,
    ) -> ModelResponse:
        observed_binary.append(binary_content_usage(messages))
        if len(observed_binary) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "attach_files",
                        {"paths": ["evidence.png"]},
                        "attach-call",
                    )
                ]
            )
        if len(observed_binary) == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "read_file",
                        {"path": "note.txt"},
                        "read-call",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    deps = AgentContext(
        app=None,
        principal=Principal("user", "tenant", "local_trusted"),
        workspace=workspace,
        session_id=None,
        execution_id="execution",
        session_metadata={},
    )
    agent = PydanticAgent(
        FunctionModel(model_function),
        deps_type=AgentContext,
        toolsets=(toolset,),
    )

    result = await agent.run(
        "inspect evidence",
        deps=deps,
        capabilities=(RuntimeCompaction(None),),
    )

    assert result.output == "done"
    assert observed_binary == [(0, 0), (1, 3), (0, 0)]
    assert binary_content_usage(result.all_messages()) == (1, 3)
