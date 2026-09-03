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
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import DisabledSandbox, SandboxSession, Workspace


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.closed = 0

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

    async def open(self) -> SandboxSession:
        session = _RecordingSession()
        self.sessions.append(session)
        return session


class _FixedSandbox:
    def __init__(self, session: _RecordingSession) -> None:
        self.session = session
        self.opens = 0

    async def open(self) -> SandboxSession:
        self.opens += 1
        return self.session


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
    workspace = Workspace.load(tmp_path)
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
        Workspace.load(tmp_path),
        Workspace.load(tmp_path, sandbox=sandbox),
        Workspace.load(tmp_path, sandbox=DisabledSandbox()),
    )
    projected = tuple(
        tuple((item.id, item.fingerprint, item.semantic_contract) for item in workspace_tool_contributions(workspace))
        for workspace in workspaces
    )
    assert projected[0] == projected[1] == projected[2]
    assert sandbox.sessions == []


def test_workspace_capabilities_materialize_one_sandbox_group(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)

    capabilities = workspace_capabilities(
        workspace,
        ("read_file", "run_command"),
    )

    assert tuple(capability.id for capability in capabilities) == ("workspace-sandbox",)
    assert workspace_capabilities(workspace, ()) == ()


def test_workspace_capabilities_with_no_selected_tools_do_not_open_sandbox(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    assert workspace_capabilities(Workspace.load(tmp_path, sandbox=sandbox), ()) == ()
    assert sandbox.sessions == []


def test_workspace_capabilities_reject_unknown_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown workspace tools"):
        workspace_capabilities(Workspace.load(tmp_path), ("missing_tool",))


@pytest.mark.asyncio
async def test_workspace_runtime_tool_semantics_match_durable_contributions(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, sandbox=sandbox)
    contributions = workspace_tool_contributions(workspace)
    expected = {item.id: item.semantic_contract for item in contributions}
    capability = workspace_capabilities(
        workspace,
        (*WORKSPACE_FILESYSTEM_TOOL_NAMES, *WORKSPACE_SHELL_TOOL_NAMES),
    )[0]
    toolset = capability.toolset  # type: ignore[attr-defined]
    run_toolset = await toolset.for_run(None)  # type: ignore[arg-type]

    assert {
        name: _semantic_contract(tool)
        for name, tool in run_toolset.tools.items()  # type: ignore[attr-defined]
    } == expected
    await run_toolset.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_workspace_sandbox_provisions_distinct_sessions_per_run(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, sandbox=sandbox)
    capability = workspace_capabilities(
        workspace,
        ("read_file", "start_command", "check_command", "stop_command"),
    )[0]
    toolset = capability.toolset  # type: ignore[attr-defined]

    first, second = await asyncio.gather(
        toolset.for_run(None),  # type: ignore[arg-type]
        toolset.for_run(None),  # type: ignore[arg-type]
    )

    assert len(sandbox.sessions) == 2
    assert sandbox.sessions[0] is not sandbox.sessions[1]
    await first.tools["read_file"].function("sample.txt")  # type: ignore[attr-defined]
    await first.tools["start_command"].function("echo one")  # type: ignore[attr-defined]
    await first.tools["check_command"].function("command")  # type: ignore[attr-defined]
    await first.tools["stop_command"].function("command")  # type: ignore[attr-defined]
    assert [name for name, _, _ in sandbox.sessions[0].calls] == [
        "read_file",
        "start_command",
        "check_command",
        "stop_command",
    ]
    assert sandbox.sessions[1].calls == []
    await first.__aexit__(None, None, None)
    await second.__aexit__(None, None, None)
    assert [session.closed for session in sandbox.sessions] == [1, 1]


@pytest.mark.asyncio
async def test_custom_sandbox_does_not_fallback_to_host_filesystem(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, sandbox=sandbox)
    capability = workspace_capabilities(workspace, ("write_file",))[0]
    toolset = capability.toolset  # type: ignore[attr-defined]
    run_toolset = await toolset.for_run(None)  # type: ignore[arg-type]

    await run_toolset.tools["write_file"].function("host.txt", "content")  # type: ignore[attr-defined]
    assert not (tmp_path / "host.txt").exists()
    assert sandbox.sessions[0].calls == [
        ("write_file", ("host.txt", "content"), {"expected_hash": None})
    ]
    await run_toolset.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_workspace_sandbox_close_failure_propagates_without_primary_error(tmp_path: Path) -> None:
    session = _FailingCloseSession()
    sandbox = _FixedSandbox(session)
    capability = workspace_capabilities(
        Workspace.load(tmp_path, sandbox=sandbox),
        ("read_file",),
    )[0]
    run_toolset = await capability.toolset.for_run(None)  # type: ignore[attr-defined,arg-type]

    with pytest.raises(RuntimeError, match="close failed"):
        await run_toolset.__aexit__(None, None, None)
    assert session.closed == 1


@pytest.mark.asyncio
async def test_workspace_sandbox_close_failure_does_not_replace_primary_error(tmp_path: Path) -> None:
    session = _FailingCloseSession()
    sandbox = _FixedSandbox(session)
    capability = workspace_capabilities(
        Workspace.load(tmp_path, sandbox=sandbox),
        ("read_file",),
    )[0]
    run_toolset = await capability.toolset.for_run(None)  # type: ignore[attr-defined,arg-type]
    primary = RuntimeError("primary")

    assert await run_toolset.__aexit__(RuntimeError, primary, None) is None
    assert session.closed == 1


@pytest.mark.asyncio
async def test_workspace_sandbox_close_is_completed_during_cancellation(tmp_path: Path) -> None:
    session = _BlockingCloseSession()
    sandbox = _FixedSandbox(session)
    capability = workspace_capabilities(
        Workspace.load(tmp_path, sandbox=sandbox),
        ("read_file",),
    )[0]
    run_toolset = await capability.toolset.for_run(None)  # type: ignore[attr-defined,arg-type]
    closing = asyncio.create_task(run_toolset.__aexit__(None, None, None))
    await session.close_started.wait()

    closing.cancel()
    await asyncio.sleep(0)
    session.close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert session.closed == 1


@pytest.mark.asyncio
async def test_disabled_sandbox_fails_before_workspace_tool_execution(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, sandbox=DisabledSandbox())
    capability = workspace_capabilities(workspace, ("read_file",))[0]
    toolset = capability.toolset  # type: ignore[attr-defined]

    with pytest.raises(AIError) as raised:
        await toolset.for_run(None)  # type: ignore[arg-type]
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
