#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-correctable failures must preserve effect certainty."""

from pathlib import Path
from typing import Any

from ._runtime_test_helpers import FileToolsStub

import pytest
from ._runtime_test_helpers import tool_run_context

from linktools.ai.capability import ToolCallRetry, filesystem_capabilities
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    BoundaryToolset,
)
from linktools.ai.workspace import LocalWorkspaceBackend, Workspace
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.decision = ToolOperationDecision("operation", "owner", 1, False)

    async def effective_args(self, ctx, call, tool_def, args):
        del ctx, call, tool_def
        return args

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: object,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args
        assert replay_safe is False
        self.calls.append("begin")
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(
        self, decision: ToolOperationDecision, error: BaseException
    ) -> None:
        del decision, error
        self.calls.append("unknown")

    async def existing_call_ids(self, tool_call_ids: tuple[str, ...]) -> frozenset[str]:
        del tool_call_ids
        return frozenset()

    async def list_operations(self) -> tuple[object, ...]:
        return ()


class _PartialWriteSession(FileToolsStub):
    def __init__(self, root: Path) -> None:
        self._root = root

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def write_file(self, path: str, content: str, *, old_text=None, expected_hash=None) -> str:
        del path
        (self._root / "partial").write_text(content)
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)


class _MissingReadSession(FileToolsStub):
    async def inspect_file(self, path: str, *, max_bytes=None) -> dict:
        del path, max_bytes
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)


@pytest.mark.asyncio
async def test_invalid_write_parent_is_known_failure_not_effect_unknown(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path)
    session = LocalWorkspaceBackend(workspace.root)
    await session.working_dir()
    try:
        workspace_capability = filesystem_capabilities(
            workspace,
            ("write_file",),
            backend=session,
        )[0]
        bridge = _Bridge()
        boundary = BoundaryToolset(
            (workspace_capability.get_toolset(),),
            {
                "write_file": ManagedToolDescriptor(
                    effect_owner="tool_operation",
                    effect_policy="non_replay_safe",
                    tool_class="filesystem.write",
                    workspace_path_fields=("path",),
                )
            },
            id="workspace-boundary",
            workspace_backend=session,
            tool_operations=bridge,  # type: ignore[arg-type]
        )
        context = tool_run_context()
        (tmp_path / "parent").write_text("not a directory")
        args = {"path": "parent/report.txt", "content": "report"}
        with pytest.raises(ToolCallRetry):
            tools = await boundary.get_tools(context)
            await boundary.call_tool("write_file", args, context, tools["write_file"])

        assert bridge.calls == ["begin", "fail"]
        assert (tmp_path / "parent").read_text() == "not a directory"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_effectful_plain_ai_error_after_partial_effect_becomes_unknown(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path)
    session = _PartialWriteSession(tmp_path)
    workspace_capability = filesystem_capabilities(
        workspace,
        ("write_file",),
        backend=session,  # type: ignore[arg-type]
    )[0]
    bridge = _Bridge()
    boundary = BoundaryToolset(
        (workspace_capability.get_toolset(),),
        {
            "write_file": ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect_policy="non_replay_safe",
                tool_class="filesystem.write",
                workspace_path_fields=("path",),
            )
        },
        id="workspace-boundary",
        workspace_backend=session,  # type: ignore[arg-type]
        tool_operations=bridge,  # type: ignore[arg-type]
    )

    context = tool_run_context()
    tools = await boundary.get_tools(context)
    with pytest.raises(AIError) as captured:
        await boundary.call_tool(
            "write_file",
            {"path": "target", "content": "partial"},
            context,
            tools["write_file"],
        )

    assert captured.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert bridge.calls == ["begin", "unknown"]
    assert (tmp_path / "partial").read_text() == "partial"


@pytest.mark.asyncio
async def test_effect_free_missing_target_remains_model_correctable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path)
    capability = filesystem_capabilities(
        workspace,
        ("read_file",),
        backend=_MissingReadSession(),  # type: ignore[arg-type]
    )[0]
    toolset = capability.get_toolset()
    context = tool_run_context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallRetry):
        await toolset.call_tool(
            "read_file",
            {"path": "missing.txt"},
            context,
            tools["read_file"],
        )
