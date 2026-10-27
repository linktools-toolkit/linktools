#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end transient file behavior across a Pydantic agent loop."""

from pathlib import Path

from ._runtime_test_helpers import FileToolsStub

import pytest
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import AgentContext, filesystem_capabilities
from linktools.ai.core import Principal, PromptLimits
from linktools.ai.runtime._compaction import CompactionCapability
from linktools.ai.model.messages import binary_content_usage
from linktools.ai.workspace import Workspace


class _Session(FileToolsStub):
    async def canonicalize_path(self, path: str) -> str:
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        values = {"evidence.png": b"png", "second.png": b"two"}
        assert path in values
        value = values[path]
        assert max_bytes is None or len(value) <= max_bytes
        return value

    async def inspect_file(
        self, path: str, *, max_bytes: int | None = None,
    ) -> dict[str, object]:
        assert path == "note.txt"
        assert max_bytes is None or max_bytes >= 4
        return {
            "path": path, "is_text": True, "mime_type": "text/plain",
            "type_source": "content", "size": 4, "state": "note-v1", "bytes_read": 4,
        }

    async def read_text(
        self,
        path: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        full_metadata: bool = False,
        max_bytes: int | None = None,
        max_read_bytes: int | None = None,
    ) -> dict[str, object]:
        assert path == "note.txt"
        assert offset is None
        assert limit is None
        assert cursor is None
        assert not full_metadata
        assert max_bytes is None or max_bytes >= 4
        assert max_read_bytes is None or max_read_bytes >= 4
        return {
            "path": path, "content": "note", "state": "note-v1", "bytes_read": 4,
            "size": 4, "offset": 0, "end_offset": 1,
            "byte_offset": 0, "end_byte_offset": 4,
            "line_offset": 0, "end_line_offset": 0,
            "lossy": False, "truncated": False, "eof": True, "next_cursor": None,
        }

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_attach_files_is_visible_for_one_model_request_only(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    session = _Session()
    toolset = filesystem_capabilities(
        workspace,
        ("attach_files", "read_file"),
        backend=session,  # type: ignore[arg-type]
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
        namespace="default",
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
        capabilities=(CompactionCapability(None, limits=PromptLimits()),),
    )

    assert result.output == "done"
    assert observed_binary == [(0, 0), (1, 3), (0, 0)]
    assert binary_content_usage(result.all_messages()) == (1, 3)


@pytest.mark.asyncio
async def test_sequential_attach_files_do_not_accumulate_binary_context(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path)
    session = _Session()
    toolset = filesystem_capabilities(
        workspace,
        ("attach_files",),
        backend=session,  # type: ignore[arg-type]
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
                        "attach-first",
                    )
                ]
            )
        if len(observed_binary) == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "attach_files",
                        {"paths": ["second.png"]},
                        "attach-second",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    deps = AgentContext(
        app=None,
        principal=Principal("user", "tenant", "local_trusted"),
        namespace="default",
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
        "inspect both files",
        deps=deps,
        capabilities=(CompactionCapability(None, limits=PromptLimits()),),
    )

    assert result.output == "done"
    assert observed_binary == [(0, 0), (1, 3), (1, 3)]
    assert binary_content_usage(result.all_messages()) == (2, 6)
