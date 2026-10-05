#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session tool recovery across process termination with SQL state."""

from pathlib import Path

import pytest

from ._session_tool_test_helpers import (
    _assert_tool_turn_recovers_without_replaying_effect,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    (
        "tool_completed",
        pytest.param("tool_checkpoint", marks=pytest.mark.merge),
        pytest.param("projected_tool_checkpoint", marks=pytest.mark.merge),
    ),
)
async def test_session_tool_turn_recovers_after_process_exit_without_replaying_effect(
    tmp_path: Path,
    phase: str,
) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(tmp_path, "sql", phase)
