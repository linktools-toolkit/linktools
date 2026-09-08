#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace model-facing contract regressions."""

import json
from pathlib import Path

import pytest
from linktools.ai.capability import workspace_capabilities, workspace_tool_contributions
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalSandbox, Workspace
from linktools.ai.workspace._sandbox_protocol import (
    MAX_FRAME_BYTES,
    encode_frame,
    validate_request_size,
)
from pydantic_ai.exceptions import ModelRetry


def _golden_contract() -> dict[str, object]:
    path = Path(__file__).parent / "fixtures" / "persistence" / "workspace_tool_semantics_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_workspace_tool_semantics_match_frozen_golden(tmp_path: Path) -> None:
    contributions = workspace_tool_contributions(Workspace.load(tmp_path))
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
    workspace = Workspace.load(tmp_path)
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("read_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ModelRetry, match="does not exist"):
            await toolset.tools["read_file"].function("missing.txt")  # type: ignore[attr-defined]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_pre_effect_write_failure_is_model_retry(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("write_file",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ModelRetry, match="parent directory"):
            await toolset.tools["write_file"].function(  # type: ignore[attr-defined]
                "missing/report.txt",
                "report",
            )
        assert not (tmp_path / "missing").exists()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_workspace_denied_shell_command_is_model_retry(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    session = await LocalSandbox().open(root=workspace.root)
    try:
        capability = workspace_capabilities(
            workspace,
            ("run_command",),
            session=session,
        )[0]
        toolset = capability.get_toolset()
        with pytest.raises(ModelRetry, match="not allowed"):
            await toolset.tools["run_command"].function("ssh example.invalid")  # type: ignore[attr-defined]
    finally:
        await session.close()
