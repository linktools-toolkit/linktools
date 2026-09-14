#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Session timeline coverage through the public Runtime API."""

from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.state import RuntimeState

from .test_runtime_composition_regressions import (
    _RuntimeUsageModels,
    _runtime_usage_workspace,
)


@pytest.mark.asyncio
async def test_in_memory_session_run_restores_timeline(tmp_path: Path) -> None:
    workspace = _runtime_usage_workspace(tmp_path / "workspace")

    async with Runtime.open(
        workspace,
        models=_RuntimeUsageModels(),  # type: ignore[arg-type]
        state=RuntimeState.in_memory(),
    ) as runtime:
        session = await runtime.agent("default").create_session("session")
        result = await session.run("hello", timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED

        page = await session.timeline()
        assert len(page.items) == 1
        turn = page.items[0]
        assert turn.execution_id == result.execution_id
        assert turn.status is ExecutionStatus.SUCCEEDED
        assert turn.user_input == {
            "version": 1,
            "prompt": {"kind": "text", "text": "hello"},
            "files": [],
        }
        assert turn.conversation_committed is True
        assert [item.item_kind for item in turn.items] == ["assistant"]
        assert page.next_cursor is None
