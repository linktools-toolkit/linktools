#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace capability projection and materialization contracts."""

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


def test_workspace_capabilities_materialize_one_sandbox_group(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)

    capabilities = workspace_capabilities(
        workspace,
        ("read_file", "run_command"),
    )

    assert tuple(capability.id for capability in capabilities) == ("workspace-sandbox",)
    assert workspace_capabilities(workspace, ()) == ()


def test_workspace_capabilities_reject_unknown_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown workspace tools"):
        workspace_capabilities(Workspace.load(tmp_path), ("missing_tool",))


@pytest.mark.asyncio
async def test_workspace_sandbox_provisions_once_per_run_and_closes_once(tmp_path: Path) -> None:
    sandbox = _RecordingSandbox()
    workspace = Workspace.load(tmp_path, sandbox=sandbox)
    capability = workspace_capabilities(workspace, ("read_file", "run_command"))[0]
    toolset = capability.toolset  # type: ignore[attr-defined]

    first = await toolset.for_run(None)  # type: ignore[arg-type]
    second = await toolset.for_run(None)  # type: ignore[arg-type]

    assert len(sandbox.sessions) == 2
    assert sandbox.sessions[0] is not sandbox.sessions[1]
    assert set(first.tools) == {"read_file", "run_command"}  # type: ignore[attr-defined]
    await first.tools["read_file"].function("sample.txt")  # type: ignore[attr-defined]
    assert sandbox.sessions[0].calls == [("read_file", ("sample.txt",), {"offset": 0, "limit": None})]
    await first.__aexit__(None, None, None)
    await second.__aexit__(None, None, None)
    assert [session.closed for session in sandbox.sessions] == [1, 1]


@pytest.mark.asyncio
async def test_disabled_sandbox_fails_before_workspace_tool_execution(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, sandbox=DisabledSandbox())
    capability = workspace_capabilities(workspace, ("read_file",))[0]
    toolset = capability.toolset  # type: ignore[attr-defined]

    with pytest.raises(AIError) as raised:
        await toolset.for_run(None)  # type: ignore[arg-type]
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
