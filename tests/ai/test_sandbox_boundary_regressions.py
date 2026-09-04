#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox lifecycle boundary regressions."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


class _FakeRunShellToolset:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def __aenter__(self) -> "_FakeRunShellToolset":
        self._events.append("enter")
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._events.append("exit")

    async def get_tools(self, ctx: object) -> dict[str, object]:
        del ctx
        self._events.append("get_tools")
        return {"run_command": object()}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, object],
        ctx: object,
        tool: object,
    ) -> str:
        del tool_args, ctx, tool
        self._events.append(("call", name))
        return "ok"


class _FakeBaseShellToolset:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def for_run(self, ctx: object) -> _FakeRunShellToolset:
        del ctx
        self._events.append("for_run")
        return _FakeRunShellToolset(self._events)


class _FakeShell:
    events: list[object] = []

    @classmethod
    def __class_getitem__(cls, item: object) -> type["_FakeShell"]:
        del item
        return cls

    def __init__(self, **kwargs: object) -> None:
        del kwargs

    def get_toolset(self) -> _FakeBaseShellToolset:
        return _FakeBaseShellToolset(self.events)


@pytest.mark.asyncio
async def test_local_sandbox_uses_harness_per_run_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.capability import _workspace

    _FakeShell.events = []
    monkeypatch.setattr(_workspace, "Shell", _FakeShell)
    capability = workspace_capabilities(Workspace.load(tmp_path), ("run_command",))[0]
    toolset = capability.toolset  # type: ignore[attr-defined]
    run_toolset = await toolset.for_run(_context())  # type: ignore[arg-type]
    try:
        result = await run_toolset.tools["run_command"].function("echo ok")  # type: ignore[attr-defined]
    finally:
        await run_toolset.__aexit__(None, None, None)

    assert result == "ok"
    assert _FakeShell.events == [
        "for_run",
        "enter",
        "get_tools",
        ("call", "run_command"),
        "exit",
    ]


class _CancellingCloseSession:
    async def close(self) -> None:
        raise asyncio.CancelledError


class _CancellingCloseSandbox:
    async def open(self) -> _CancellingCloseSession:
        return _CancellingCloseSession()


@pytest.mark.asyncio
async def test_cancelled_close_does_not_replace_primary_failure(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, sandbox=_CancellingCloseSandbox())  # type: ignore[arg-type]
    capability = workspace_capabilities(workspace, ("read_file",))[0]
    run_toolset = await capability.toolset.for_run(_context())  # type: ignore[attr-defined,arg-type]
    primary = RuntimeError("primary")

    assert await run_toolset.__aexit__(RuntimeError, primary, None) is None


@pytest.mark.asyncio
async def test_cancelled_close_propagates_without_primary_failure(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, sandbox=_CancellingCloseSandbox())  # type: ignore[arg-type]
    capability = workspace_capabilities(workspace, ("read_file",))[0]
    run_toolset = await capability.toolset.for_run(_context())  # type: ignore[attr-defined,arg-type]

    with pytest.raises(asyncio.CancelledError):
        await run_toolset.__aexit__(None, None, None)
