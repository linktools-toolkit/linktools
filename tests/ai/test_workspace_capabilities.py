#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace capability projection and materialization contracts."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.capability import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    CapabilityGroup,
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import (
    DisabledSandbox,
    RepositoryInstructions,
    SandboxResource,
    SandboxSession,
    ToolPermissionRule,
    Workspace,
    WorkspaceToolPermissionPolicy,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ApprovalRequired, ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.closed = 0

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def _record(self, name: str, *args: object, **kwargs: object) -> str:
        self.calls.append((name, args, kwargs))
        return name

    async def read_file(self, path: str, *, offset: int = 0, limit: "int | None" = None) -> str:
        return await self._record("read_file", path, offset=offset, limit=limit)

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        return await self._record("write_file", path, content, expected_hash=expected_hash)

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        return await self._record(
            "edit_file",
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    async def list_directory(self, path: str = ".") -> str:
        return await self._record("list_directory", path)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: "str | None" = None,
    ) -> str:
        return await self._record(
            "search_files",
            pattern,
            path=path,
            include_glob=include_glob,
        )

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        return await self._record("find_files", pattern, path=path)

    async def create_directory(self, path: str) -> str:
        return await self._record("create_directory", path)

    async def file_info(self, path: str) -> str:
        return await self._record("file_info", path)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str:
        return await self._record("run_command", command, timeout_seconds=timeout_seconds)

    async def start_command(self, command: str) -> str:
        return await self._record("start_command", command)

    async def check_command(self, command_id: str) -> str:
        return await self._record("check_command", command_id)

    async def stop_command(self, command_id: str) -> str:
        return await self._record("stop_command", command_id)

    async def close(self) -> None:
        self.closed += 1


class _RecordingSandbox:
    def __init__(self) -> None:
        self.sessions: list[_RecordingSession] = []

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        session = _RecordingSession()
        self.sessions.append(session)
        return session


class _FailingCloseSession(_RecordingSession):
    async def close(self) -> None:
        self.closed += 1
        raise RuntimeError("close failed")


class _BlockingCloseSession(_RecordingSession):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()

    async def close(self) -> None:
        self.closed += 1
        self.close_started.set()
        await self.close_release.wait()


class _UnusedResolver:
    async def resolve(
        self,
        target: str,
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        del target, exclude_sources
        return RepositoryInstructions(())


class _SpoofedSandboxCapability(AbstractCapability[object]):
    id = "linktools.workspace-sandbox"


def _semantic_contract(tool: object) -> dict[str, object]:
    definition = tool.tool_def  # type: ignore[attr-defined]
    return {
        "version": 1,
        "description": definition.description,
        "parameters": definition.parameters_json_schema,
        "return_schema": definition.return_schema,
        "strict": definition.strict,
        "metadata": definition.metadata,
    }


def test_workspace_tool_contributions_are_stable_and_classified(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    contributions = workspace_tool_contributions(workspace)

    assert tuple(item.id for item in contributions) == (
        *WORKSPACE_FILESYSTEM_TOOL_NAMES,
        *WORKSPACE_SHELL_TOOL_NAMES,
    )
    assert all(item.kind == "tool" for item in contributions)
    assert all(len(item.fingerprint) == 64 for item in contributions)
    assert tuple(workspace_tool_class(item.value) for item in contributions) == tuple(
        "filesystem.read"
        if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        else "filesystem.write"
        for name in WORKSPACE_FILESYSTEM_TOOL_NAMES
    ) + tuple("shell" for _ in WORKSPACE_SHELL_TOOL_NAMES)
    assert tuple(item.fingerprint for item in contributions) == tuple(
        item.fingerprint for item in workspace_tool_contributions(workspace)
    )


def test_workspace_tool_declarations_do_not_depend_on_sandbox_selection(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspaces = (
        Workspace.load(tmp_path, workspace_id="workspace"),
        Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox),
        Workspace.load(tmp_path, workspace_id="workspace", sandbox=DisabledSandbox()),
    )
    projected = tuple(
        tuple((item.id, item.fingerprint, item.semantic_contract) for item in workspace_tool_contributions(workspace))
        for workspace in workspaces
    )
    assert projected[0] == projected[1] == projected[2]
    assert sandbox.sessions == []


def test_workspace_capabilities_materialize_one_sandbox_group(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")

    with pytest.raises(AIError) as raised:
        workspace_capabilities(workspace, ("read_file", "run_command"))
    assert raised.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
    assert workspace_capabilities(workspace, ()) == ()


def test_workspace_capabilities_with_no_selected_tools_do_not_open_sandbox(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    assert workspace_capabilities(Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox), ()) == ()
    assert sandbox.sessions == []


def test_workspace_capabilities_reject_unknown_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown workspace tools"):
        workspace_capabilities(Workspace.load(tmp_path, workspace_id="workspace"), ("missing_tool",))


def test_workspace_sandbox_capability_id_is_reserved() -> None:
    group = CapabilityGroup[object]("custom")
    with pytest.raises(AIError) as raised:
        group.capability(_SpoofedSandboxCapability())
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_workspace_runtime_tool_semantics_match_durable_contributions(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox)
    contributions = workspace_tool_contributions(workspace)
    expected = {item.id: item.semantic_contract for item in contributions}
    capability = workspace_capabilities(
        workspace,
        (*WORKSPACE_FILESYSTEM_TOOL_NAMES, *WORKSPACE_SHELL_TOOL_NAMES),
        session=await sandbox.open(root=workspace.root),
    )[0]
    run_toolset = capability.get_toolset()

    assert {
        name: _semantic_contract(tool)
        for name, tool in run_toolset.tools.items()  # type: ignore[attr-defined]
    } == expected
    await run_toolset.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_workspace_capability_uses_the_caller_owned_session(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox)
    capability = workspace_capabilities(
        workspace,
        ("read_file", "start_command", "check_command", "stop_command"),
        session=await sandbox.open(root=workspace.root),
    )[0]
    toolset = capability.get_toolset()

    assert len(sandbox.sessions) == 1
    await toolset.tools["read_file"].function("sample.txt")  # type: ignore[attr-defined]
    await toolset.tools["start_command"].function("echo one")  # type: ignore[attr-defined]
    await toolset.tools["check_command"].function("command")  # type: ignore[attr-defined]
    await toolset.tools["stop_command"].function("command")  # type: ignore[attr-defined]
    assert [name for name, _, _ in sandbox.sessions[0].calls] == [
        "read_file",
        "start_command",
        "check_command",
        "stop_command",
    ]
    await toolset.__aexit__(None, None, None)
    await sandbox.sessions[0].close()
    assert [session.closed for session in sandbox.sessions] == [1]


@pytest.mark.parametrize(
    ("decision", "expected_error"),
    (("deny", ToolFailed), ("ask", ApprovalRequired)),
)
@pytest.mark.asyncio
async def test_permission_rejection_has_no_sandbox_operation_side_effect(
    tmp_path: Path,
    decision: str,
    expected_error: type[BaseException],
) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox)
    session = await sandbox.open(root=workspace.root)
    capability = workspace_capabilities(
        workspace,
        ("read_file",),
        session=session,
    )[0]
    run_toolset = capability.get_toolset()
    boundary = RuntimeToolBoundaryToolset(
        (run_toolset,),
        {
            "read_file": ManagedToolDescriptor(
                effect_owner="intrinsic",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace-boundary",
            workspace_policy=WorkspaceToolPermissionPolicy(
                (ToolPermissionRule(decision, tool_name="read_file"),)  # type: ignore[arg-type]
            ),
        sandbox_session=session,
    )
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )
    tools = await boundary.get_tools(context)

    with pytest.raises(expected_error):
        await boundary.call_tool(
            "read_file",
            {"path": "sample.txt"},
            context,
            tools["read_file"],
        )
    assert sandbox.sessions[0].calls == []
    await run_toolset.__aexit__(None, None, None)
    await session.close()


@pytest.mark.asyncio
async def test_custom_sandbox_does_not_fallback_to_host_filesystem(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=sandbox)
    session = await sandbox.open(root=workspace.root)
    capability = workspace_capabilities(
        workspace,
        ("write_file",),
        session=session,
    )[0]
    run_toolset = capability.get_toolset()

    await run_toolset.tools["write_file"].function("host.txt", "content")  # type: ignore[attr-defined]
    assert not (tmp_path / "host.txt").exists()
    assert sandbox.sessions[0].calls == [
        ("write_file", ("host.txt", "content"), {"expected_hash": None})
    ]
    await run_toolset.__aexit__(None, None, None)
    await session.close()


@pytest.mark.asyncio
async def test_workspace_sandbox_close_failure_propagates_without_primary_error(tmp_path: Path) -> None:
    session = _FailingCloseSession()

    with pytest.raises(RuntimeError, match="close failed"):
        await session.close()
    assert session.closed == 1


@pytest.mark.asyncio
async def test_workspace_sandbox_close_is_completed_during_cancellation(tmp_path: Path) -> None:
    session = _BlockingCloseSession()
    closing = asyncio.create_task(session.close())
    await session.close_started.wait()

    closing.cancel()
    await asyncio.sleep(0)
    session.close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert session.closed == 1


@pytest.mark.asyncio
async def test_disabled_sandbox_fails_before_workspace_tool_execution(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace", sandbox=DisabledSandbox())

    with pytest.raises(AIError) as raised:
        await workspace.sandbox.open(root=workspace.root)  # type: ignore[union-attr]
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
