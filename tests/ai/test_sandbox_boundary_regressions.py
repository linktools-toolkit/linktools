#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox lifecycle boundary regressions."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.runtime._compaction import (
    _SUMMARY_MARKER,
    _is_context_summary,
    _insert_summary,
    _protected_message_indexes,
    _summary_candidate,
)
from linktools.ai.workspace import SandboxResource, Workspace
from linktools.ai.workspace._bubblewrap import _build_bwrap_args
from pydantic_ai.messages import ModelRequest, UserPromptPart
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


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        del timeout_seconds
        self.calls.append(command)
        return "ok"


@pytest.mark.asyncio
async def test_workspace_capability_adapts_the_caller_owned_session(
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    capability = workspace_capabilities(
        Workspace.load(tmp_path),
        ("run_command",),
        session=session,  # type: ignore[arg-type]
    )[0]
    toolset = capability.get_toolset()
    tools = await toolset.get_tools(_context())
    result = await toolset.call_tool(
        "run_command",
        {"command": "echo ok"},
        _context(),
        tools["run_command"],
    )

    assert result == "ok"
    assert session.calls == ["echo ok"]


class _CancellingCloseSession:
    async def close(self) -> None:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_cancelled_close_does_not_replace_primary_failure(tmp_path: Path) -> None:
    del tmp_path
    session = _CancellingCloseSession()
    with pytest.raises(asyncio.CancelledError):
        await session.close()


@pytest.mark.asyncio
async def test_cancelled_close_propagates_without_primary_failure(tmp_path: Path) -> None:
    del tmp_path
    session = _CancellingCloseSession()
    with pytest.raises(asyncio.CancelledError):
        await session.close()


def test_context_summary_replaces_prior_summary_generations() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(f"message-{index}")])
        for index in range(30)
    ]
    for index in (8, 15, 20):
        messages[index] = ModelRequest(
            parts=[UserPromptPart(f"{_SUMMARY_MARKER}\nold-{index}")],
            metadata={"linktools.ai.context_summary": {"source_indexes": [index]}},
        )

    protected = _protected_message_indexes(messages)
    candidate = _summary_candidate(messages, protected)

    assert candidate is not None
    retained = [
        message
        for index, message in enumerate(messages)
        if index not in candidate.indices
    ]
    replacement = ModelRequest(
        parts=[UserPromptPart(f"{_SUMMARY_MARKER}\nnew")],
        metadata={"linktools.ai.context_summary": {"source_indexes": []}},
    )
    projected = _insert_summary(messages, candidate.indices, replacement)

    assert len(retained) >= 20
    assert sum(_is_context_summary(message) for message in projected) == 1


def test_context_summary_keeps_protected_standalone_instruction() -> None:
    messages = [
        ModelRequest(parts=[UserPromptPart(f"message-{index}")])
        for index in range(30)
    ]
    messages[0] = ModelRequest(
        instructions="retain this instruction",
        parts=[UserPromptPart("old context")],
    )

    protected = _protected_message_indexes(messages)
    candidate = _summary_candidate(messages, protected)

    assert candidate is not None
    assert 0 in protected
    assert 0 not in candidate.indices


def test_bubblewrap_hidden_mount_overrides_workspace_resource_bind(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "package"
    source.mkdir()
    (source / "private").mkdir()
    resource = SandboxResource("resource", source)

    arguments = _build_bwrap_args(
        root=workspace,
        runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"),
        lock_root=workspace / ".linktools" / "locks",
        resources=(resource,),
        hidden_paths=("package/private",),
        worker_resources=[{"key": "resource", "path": "/skills/resource"}],
    )

    visible_bind = arguments.index("/workspace/package")
    hidden_mount = arguments.index("/workspace/package/private")
    assert hidden_mount > visible_bind


def test_bubblewrap_allows_explicit_skill_resource_under_hidden_storage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / ".linktools" / "skills" / "review"
    source.mkdir(parents=True)
    resource = SandboxResource("resource", source)

    arguments = _build_bwrap_args(
        root=workspace,
        runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"),
        lock_root=workspace / ".linktools" / "locks",
        resources=(resource,),
        hidden_paths=(".linktools",),
        worker_resources=[{"key": "resource", "path": "/skills/resource"}],
    )

    assert arguments.count(str(source)) == 1
    assert "/workspace/.linktools/skills/review" not in arguments
