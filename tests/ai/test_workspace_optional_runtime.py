#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-optional Runtime behavior and file-source boundaries."""

from pathlib import Path

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
async def test_workspace_less_runtime_runs_session_and_history() -> None:
    async with Runtime.open(
        "web-chat",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
    ) as runtime:
        assert runtime.namespace == "web-chat"
        session = await runtime.agent("default").create_session("chat")
        result = await session.run("hello", timeout_seconds=10)

        assert result.status is ExecutionStatus.SUCCEEDED
        history = await session.history()
        assert history.items


@pytest.mark.asyncio
async def test_workspace_less_runtime_rejects_files_and_explicit_cwd() -> None:
    async with Runtime.open(
        "web-chat",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
    ) as runtime:
        with pytest.raises(AIError) as files_error:
            await runtime.agent("default").run(
                "inspect",
                files=("evidence.txt",),
                timeout_seconds=10,
            )
        assert files_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert files_error.value.safe_details == {
            "field": "files",
            "reason": "workspace_required",
        }

        with pytest.raises(AIError) as cwd_error:
            await runtime.agent("default").create_session(
                "cwd-session",
                cwd=".",
            )
        assert cwd_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert cwd_error.value.safe_details == {
            "field": "cwd",
            "reason": "workspace_required",
        }


@pytest.mark.asyncio
async def test_existing_workspace_cwd_requires_workspace_for_new_turn(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    workspace = Workspace.load(project, workspace_id="workspace")
    state_root = tmp_path / "runtime"

    async with Runtime.open(
        "workspace",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.from_root(state_root),
        capabilities=(CapabilityGroup.from_workspace(workspace),),
    ) as runtime:
        await runtime.agent("default").create_session(
            "cwd-session",
            cwd=".",
        )

    async with Runtime.open(
        "workspace",
        models=RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.from_root(state_root),
    ) as runtime:
        with pytest.raises(AIError) as error:
            await runtime.agent("default").run(
                "continue",
                session_id="cwd-session",
                timeout_seconds=10,
            )

        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert error.value.safe_details == {
            "field": "cwd",
            "reason": "workspace_required",
        }
