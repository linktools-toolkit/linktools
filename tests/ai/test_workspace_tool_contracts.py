#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-facing contract regressions."""

import json
from pathlib import Path

import pytest
from linktools.ai.capability import (
    CapabilityGroup,
    ToolCallRetry,
    workspace_capabilities,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalSandbox, Workspace


def _workspace_tool_contributions(workspace: Workspace):
    return tuple(
        CapabilityGroup.from_workspace(
            workspace,
            discover_assets=False,
        )._contributions
    )

from linktools.ai.workspace._sandbox_protocol import (
    MAX_FRAME_BYTES,
    encode_frame,
    validate_request_size,
)


def _golden_contract() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "persistence" / "workspace_tool_semantics_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_workspace_tool_semantics_match_frozen_golden(tmp_path: Path) -> None:
    contributions = _workspace_tool_contributions(Workspace.load(tmp_path, workspace_id="workspace"))
    actual = {item.id: item.semantic_contract for item in contributions}
    assert actual == _golden_contract()


def test_local_semantic_validation_is_not_limited_by_ipc_frame() -> None:
    content = "x" * (MAX_FRAME_BYTES + 1)
    params = {"path": "large.txt", "content": content, "expected_hash": None}

    validate_request_size("write_file", params)
    with pytest.raises(AIError) as raised:
        encode_frame(
            {
                "version": 1,
                "request_id": "request",
                "method": "write_file",
                "params": params,
            }
        )
    assert raised.value.code is ErrorCode.TOOL_ARGUMENTS_TOO_LARGE


@pytest.mark.asyncio
async def test_workspace_missing_read_is_model_retry(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("read_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="does not exist"):
            await toolset.tools["read_file"].function("missing.txt")  # type: ignore[attr-defined]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_invalid_read_window_explains_correction(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("line\n", encoding="utf-8")
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("read_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="non-negative offset"):
            await toolset.tools["read_file"].function(  # type: ignore[attr-defined]
                "note.txt",
                offset=-1,
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_invalid_utf8_edit_explains_text_boundary(tmp_path: Path) -> None:
    (tmp_path / "binary.txt").write_bytes(b"\xff")
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("edit_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="valid UTF-8 text"):
            await toolset.tools["edit_file"].function(  # type: ignore[attr-defined]
                "binary.txt",
                "old",
                "new",
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_edit_miss_explains_unique_match_requirement(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("before\n", encoding="utf-8")
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("edit_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="matches exactly once"):
            await toolset.tools["edit_file"].function(  # type: ignore[attr-defined]
                "note.txt",
                "missing",
                "after",
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_unknown_attachment_type_explains_alternative(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("attach_files",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="recognized media type"):
            await toolset.tools["attach_files"].function(  # type: ignore[attr-defined]
                ["evidence.unknown"]
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_unknown_command_id_explains_source(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("check_command",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="active command id"):
            await toolset.tools["check_command"].function(  # type: ignore[attr-defined]
                "missing-command"
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_pre_effect_write_failure_is_model_retry(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("write_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="parent directory"):
            await toolset.tools["write_file"].function(  # type: ignore[attr-defined]
                "missing/report.txt",
                "report",
            )
        assert not (tmp_path / "missing").exists()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_denied_shell_command_is_model_retry(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("run_command",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ToolCallRetry, match="not allowed"):
            await toolset.tools["run_command"].function("ssh example.invalid")  # type: ignore[attr-defined]
    finally:
        await session.close()
