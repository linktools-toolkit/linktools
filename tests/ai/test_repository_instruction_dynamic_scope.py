#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction checks at the final runtime tool boundary."""

from typing import Any

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RepositoryInstructionBoundary,
    BoundaryToolset,
)
from linktools.ai.capability import ToolCallFailed
from linktools.ai.workspace import (
    ToolPermissionPolicy,
)
from ._runtime_test_helpers import tool_with_metadata


class _Session:
    async def canonicalize_path(self, path: str) -> str:
        return "." if path == "" else path


class _Boundary:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def render_initial(self) -> str:
        return "repository initial instructions"

    def render_overlay(self) -> str:
        return "repository overlay instructions"

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        path_fields: tuple[str, ...],
        approved: bool = False,
    ) -> None:
        self.calls.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "path_fields": path_fields,
            }
        )
        if self.fail:
            raise ToolCallFailed("repository instructions changed")


async def _read_file(path: str) -> str:
    return path


async def _list_directory(path: str = ".") -> str:
    return path


def _context(call_id: str = "call") -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id=call_id,
    )


def _boundary(
    toolset: FunctionToolset[None],
    name: str,
    repository: RepositoryInstructionBoundary,
    *,
    policy: ToolPermissionPolicy | None = None,
    path_fields: tuple[str, ...] = ("path",),
) -> BoundaryToolset:
    descriptor = ManagedToolDescriptor(
        effect_owner="none",
        effect_policy="none",
        tool_class="filesystem.read",
        workspace_path_fields=path_fields,
    )
    toolset.tools[name].metadata = tool_with_metadata(
        toolset.tools[name].function,
        descriptor,
    ).metadata
    return BoundaryToolset(
        (toolset,),
        {name: descriptor},
        id="workspace",
        workspace_backend=_Session(),  # type: ignore[arg-type]
        permission_policy=(None if policy is None else policy),
        repository_boundary=repository,
    )


@pytest.mark.asyncio
async def test_repository_instruction_check_precedes_ask_permission() -> None:
    repository = _Boundary(fail=True)
    toolset = _boundary(
        FunctionToolset([_read_file]),
        "_read_file",
        repository,
        policy=ToolPermissionPolicy(default="ask"),
    )
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallFailed, match="repository instructions changed"):
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )

    assert len(repository.calls) == 1
    assert repository.calls[0]["arguments"] == {"path": "pkg/file.txt"}


@pytest.mark.asyncio
async def test_repository_instruction_refresh_can_fence_one_model_call() -> None:
    repository = _Boundary(fail=True)
    toolset = _boundary(FunctionToolset([_read_file]), "_read_file", repository)
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolCallFailed):
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )
    assert repository.calls[0]["tool_call_id"] == "call"


@pytest.mark.asyncio
async def test_empty_workspace_path_is_normalized_before_instruction_check() -> None:
    repository = _Boundary()
    toolset = _boundary(
        FunctionToolset([_list_directory]),
        "_list_directory",
        repository,
    )
    context = _context()
    tools = await toolset.get_tools(context)

    result = await toolset.call_tool(
        "_list_directory",
        {"path": ""},
        context,
        tools["_list_directory"],
    )

    assert result == "."
    assert repository.calls[0]["arguments"] == {"path": "."}


@pytest.mark.asyncio
async def test_approved_call_reaches_tool_after_instruction_check() -> None:
    repository = _Boundary()
    toolset = _boundary(
        FunctionToolset([_read_file]),
        "_read_file",
        repository,
        policy=ToolPermissionPolicy(default="ask"),
    )
    context = _context()
    context.tool_call_approved = True
    tools = await toolset.get_tools(context)

    assert (
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )
        == "pkg/file.txt"
    )
    assert repository.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", (None, "call", "tool", "frontier", "unapproved", "approval_flag", "new_scope"))
async def test_recovered_approval_only_reuses_its_observed_instruction_frontier(mismatch) -> None:
    from types import SimpleNamespace

    from pydantic_ai.exceptions import ApprovalRequired

    from linktools.ai.runtime._repository_instructions import (
        _RepositoryInstructionApproval,
        _RepositoryInstructionBoundary,
    )
    from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions

    class Coordinator:
        def __init__(self):
            self.checked = []
            self.overlay = None

        async def check_repository_instructions(self, **kwargs):
            self.checked.append(kwargs)
            return self.overlay, self.overlay != kwargs["overlay"]

    coordinator = Coordinator()
    execution = SimpleNamespace(execution_id="execution")
    initial = RepositoryInstructions((RepositoryInstructionDocument("agents:AGENTS.md", ".", "root"),))
    original = _RepositoryInstructionBoundary(coordinator, execution, initial, None)
    original.render_overlay()  # The request producing this call observed the frontier.
    toolset = _boundary(FunctionToolset([_read_file]), "_read_file", original,
                        policy=ToolPermissionPolicy(default="ask"))
    tools = await toolset.get_tools(_context())
    with pytest.raises(ApprovalRequired) as approval:
        await toolset.call_tool("_read_file", {"path": "file.txt"}, _context(), tools["_read_file"])
    call = ToolCallPart("_read_file", {"path": "file.txt"}, "call")
    frontier = original.approval_frontiers((call,))["call"]
    assert isinstance(frontier, str)
    assert "repository_instruction_frontier" not in approval.value.metadata["linktools"]
    if mismatch == "frontier":
        # A sibling in the SAME model response discovers instructions after A
        # has entered approval. Its newer frontier was never seen by the model.
        coordinator.overlay = RepositoryInstructions((RepositoryInstructionDocument(
            "agents:pkg/AGENTS.md", "pkg", "new sibling rule",
        ),))
        with pytest.raises(ToolCallFailed):
            await original.check(tool_name="_read_file", tool_call_id="sibling",
                                 arguments={"path": "pkg/file.txt"}, path_fields=("path",))
        assert original.approval_frontiers((call,))["call"] == frontier
    checked_before_resume = len(coordinator.checked)
    restored = _RepositoryInstructionBoundary(
        coordinator, execution, initial, coordinator.overlay,
        approved_frontiers={} if mismatch == "unapproved" else {"call": _RepositoryInstructionApproval(
            "_read_file", frontier,
        )},
    )
    restored_tools = _boundary(FunctionToolset([_read_file]), "_read_file", restored,
                              policy=ToolPermissionPolicy(default="ask"))
    context = _context("different" if mismatch == "call" else "call")
    context.tool_call_approved = mismatch != "approval_flag"
    if mismatch == "new_scope":
        coordinator.overlay = RepositoryInstructions((RepositoryInstructionDocument(
            "agents:pkg/AGENTS.md", "pkg", "new approved target rule",
        ),))
    if mismatch == "tool":
        with pytest.raises(ToolCallFailed):
            await restored.check(tool_name="another_tool", tool_call_id="call",
                                 arguments={"path": "file.txt"}, path_fields=("path",), approved=True)
    else:
        tools = await restored_tools.get_tools(context)
        args = {"path": "pkg/new.txt" if mismatch == "new_scope" else "file.txt"}
        if mismatch is None:
            assert await restored_tools.call_tool("_read_file", args, context, tools["_read_file"]) == "file.txt"
            # Approval permits observation replay, not bypassing the path check.
            assert len(coordinator.checked) == checked_before_resume + 1
        else:
            with pytest.raises(ToolCallFailed):
                await restored_tools.call_tool("_read_file", args, context, tools["_read_file"])
    if mismatch is not None:
        assert len(coordinator.checked) == checked_before_resume + (mismatch == "new_scope")
