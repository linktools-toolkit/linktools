#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-correctable failures must preserve effect certainty."""

from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import ToolCallRetry, workspace_capabilities
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import LocalSandbox, Workspace
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


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


class _PartialDirectorySession:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def create_directory(self, path: str) -> str:
        del path
        (self._root / "partial").mkdir()
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)


class _MissingReadSession:
    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        del path, offset, limit
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_missing_write_parent_is_known_failure_not_effect_unknown(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        workspace_capability = workspace_capabilities(
            workspace,
            ("write_file",),
            session=session,
        )[0]
        bridge = _Bridge()
        boundary = RuntimeToolBoundaryToolset(
            (workspace_capability.get_toolset(),),
            {
                "write_file": ManagedToolDescriptor(
                    effect_owner="tool_operation",
                    effect="non_replay_safe",
                    tool_class="filesystem.write",
                    workspace_path_fields=("path",),
                )
            },
            id="workspace-boundary",
            sandbox_session=session,
            tool_operations=bridge,  # type: ignore[arg-type]
        )
        context = _context()
        args = {"path": "missing/report.txt", "content": "report"}
        with pytest.raises(ToolCallRetry):
            tools = await boundary.get_tools(context)
            await boundary.call_tool("write_file", args, context, tools["write_file"])

        assert bridge.calls == ["begin", "fail"]
        assert not (tmp_path / "missing").exists()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_effectful_plain_ai_error_after_partial_effect_becomes_unknown(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _PartialDirectorySession(tmp_path)
    workspace_capability = workspace_capabilities(
        workspace,
        ("create_directory",),
        session=session,  # type: ignore[arg-type]
    )[0]
    bridge = _Bridge()
    boundary = RuntimeToolBoundaryToolset(
        (workspace_capability.get_toolset(),),
        {
            "create_directory": ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="filesystem.write",
                workspace_path_fields=("path",),
            )
        },
        id="workspace-boundary",
        sandbox_session=session,  # type: ignore[arg-type]
        tool_operations=bridge,  # type: ignore[arg-type]
    )

    context = _context()
    tools = await boundary.get_tools(context)
    with pytest.raises(AIError) as captured:
        await boundary.call_tool(
            "create_directory",
            {"path": "target"},
            context,
            tools["create_directory"],
        )

    assert captured.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert bridge.calls == ["begin", "unknown"]
    assert (tmp_path / "partial").is_dir()


@pytest.mark.asyncio
async def test_effect_free_missing_target_remains_model_correctable(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    capability = workspace_capabilities(
        workspace,
        ("read_file",),
        session=_MissingReadSession(),  # type: ignore[arg-type]
    )[0]
    toolset = capability.get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallRetry):
        await toolset.call_tool(
            "read_file",
            {"path": "missing.txt"},
            context,
            tools["read_file"],
        )
