#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit Workspace registration, per-run binding, and listing semantics."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import RuntimeFeatures, WorkspaceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.workspace import LocalSandbox, Workspace

from .test_explicit_runtime_features import _FeatureModels


@pytest.mark.asyncio
async def test_workspace_capability_is_the_only_workspace_registration(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    features = RuntimeFeatures("workspace")
    assert (await features.capture()).workspace is None
    with pytest.raises(TypeError):
        RuntimeFeatures("workspace", workspace=workspace)
    capability = WorkspaceCapability(workspace)
    with pytest.raises(AIError) as invalid:
        features.capability(capability, scope="agent")
    assert invalid.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    assert not (await features.capture()).contributions
    features.capability(capability, scope="runtime")
    capture = await features.capture()
    assert capture.workspace is workspace
    assert {item.id for item in capture.contributions if item.kind == "tool"} == {
        tool.name for tool in capability.tool_definitions()
    }
    with pytest.raises(AIError) as duplicate:
        features.capability(WorkspaceCapability(workspace), scope="runtime")
    assert duplicate.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.asyncio
async def test_workspace_bindings_are_isolated_and_absence_keeps_resource(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    capability = WorkspaceCapability(workspace)
    first = await LocalSandbox().open(root=tmp_path)
    second = await LocalSandbox().open(root=tmp_path)
    try:
        one, two = await asyncio.gather(
            asyncio.to_thread(capability.bind, ("list_files",), session=first),
            asyncio.to_thread(capability.bind, ("list_directory",), session=second),
        )
        assert one is not two and one is not capability
        assert set(one.get_toolset().tools) == {"list_files"}
        assert set(two.get_toolset().tools) == {"list_directory"}
        empty = capability.bind((), session=None)
        assert empty.workspace is workspace
        assert empty.get_toolset() is None and empty.get_instructions() is None
        with pytest.raises(AIError) as unbound:
            capability.get_toolset()
        assert unbound.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert (await one.get_toolset().tools["list_files"].function()) == "(empty directory)"
        assert two.get_instructions() is not None
    finally:
        await asyncio.gather(first.close(), second.close())


@pytest.mark.asyncio
async def test_native_wrapped_workspace_has_one_selected_tool_per_run(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    template = WorkspaceCapability(Workspace.load(tmp_path))
    features = RuntimeFeatures("wrapped")
    features.capability(CombinedCapability([template.prefix_tools("prefix")], id="workspace.bundle"), scope="runtime")
    features.agent("default", allow_tools=("list_files",), allow_capabilities=())
    models = _FeatureModels(TestModel(call_tools=["prefix_list_files"]))
    async with Runtime.open("workspace-native", models=models, storage=RuntimeStorage.in_memory(), features=features) as runtime:
        first, second = await asyncio.gather(
            runtime.agents.get().run("list files", timeout_seconds=10),
            runtime.agents.get().run("list files", timeout_seconds=10),
        )
        for result in (first.result, second.result):
            assert result.status.value == "SUCCEEDED"
            assert "hello.txt" in str(result.output)
    with pytest.raises(AIError):
        template.get_toolset()


@pytest.mark.asyncio
async def test_list_files_omits_unknown_mime_without_hiding_entries(tmp_path: Path) -> None:
    (tmp_path / "known.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "untyped.zzz_no_known_mime").write_bytes(b"\x00\x01")
    (tmp_path / "directory").mkdir()
    session = await LocalSandbox().open(root=tmp_path)
    try:
        capability = WorkspaceCapability(Workspace.load(tmp_path)).bind(("list_files",), session=session)
        listing = await capability.get_toolset().tools["list_files"].function()
        rows = {row.split("  ")[0]: row.split("  ") for row in listing.splitlines()}
        assert rows["known.txt"] == ["known.txt", "file", "5", "text/plain", "extension"]
        assert rows["untyped.zzz_no_known_mime"] == ["untyped.zzz_no_known_mime", "file", "2"]
        assert rows["directory"][1] == "directory" and len(rows["directory"]) == 3
        assert "unknown" not in listing and "application/octet-stream" not in listing
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ("deny", "ask"))
async def test_prefixed_workspace_write_still_checks_original_tool_policy(tmp_path: Path, decision: str) -> None:
    from pydantic_ai.exceptions import ApprovalRequired
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage
    from linktools.ai.capability import ToolCallFailed
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, managed_tool_descriptor_from_metadata
    from linktools.ai.workspace import ToolPermissionPolicy, ToolPermissionRule

    session = await LocalSandbox().open(root=tmp_path)
    try:
        template = WorkspaceCapability(Workspace.load(tmp_path))
        declarations = {tool.name: tool for tool in template.tool_definitions()}
        def wrap(toolset):
            return BoundaryToolset(
                (toolset,), {"write_file": managed_tool_descriptor_from_metadata(declarations["write_file"].metadata)},
                id="workspace-policy", sandbox_session=session,
                permission_policy=ToolPermissionPolicy((ToolPermissionRule(decision, tool_name="write_file"),)),
            )
        toolset = template.bind(("write_file",), session=session, wrap_toolset=wrap).prefix_tools("files").get_toolset()
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run", tool_call_id="call")
        tools = await toolset.get_tools(ctx)
        with pytest.raises(ToolCallFailed if decision == "deny" else ApprovalRequired) as denied:
            await toolset.call_tool("files_write_file", {"path": "new.txt", "content": "must not write"}, ctx, tools["files_write_file"])
        assert not (tmp_path / "new.txt").exists()
        if decision == "ask":
            assert denied.value.metadata["linktools"]["canonical_args"]["path"] == "new.txt"
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cycle", (False, True))
async def test_workspace_boundary_rejects_unrelated_or_cyclic_wrapper(tmp_path: Path, cycle: bool) -> None:
    from dataclasses import replace
    from pydantic_ai.tools import RunContext
    from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
    from pydantic_ai.usage import RunUsage
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, managed_tool_descriptor_from_metadata

    session = await LocalSandbox().open(root=tmp_path)
    try:
        template = WorkspaceCapability(Workspace.load(tmp_path))
        raw = template.bind(("list_files",), session=session).get_toolset()
        declaration = next(tool for tool in template.tool_definitions() if tool.name == "list_files")
        boundary = BoundaryToolset((raw,), {"list_files": managed_tool_descriptor_from_metadata(declaration.metadata)}, id="workspace-owner", sandbox_session=session)
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run", tool_call_id="call")
        tools = await boundary.get_tools(ctx)
        foreign = WrapperToolset(FunctionToolset())
        if cycle:
            foreign.wrapped = foreign
        with pytest.raises(AIError) as denied:
            await boundary.call_tool("list_files", {}, ctx, replace(tools["list_files"], toolset=foreign))
        assert denied.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await session.close()
