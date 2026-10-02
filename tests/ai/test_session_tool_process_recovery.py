#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session tool result recovery across process termination boundaries."""

from pathlib import Path

import pytest

from ._session_tool_test_helpers import (
    _assert_tool_turn_recovers_without_replaying_effect,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql", "split_sqlite"))
@pytest.mark.parametrize(
    "phase",
    ("tool_completed", "tool_checkpoint", "projected_tool_checkpoint"),
)
async def test_session_tool_turn_recovers_after_process_exit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(tmp_path, backend, phase)
