#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session tool result recovery around terminal commit process exits."""

from pathlib import Path

import pytest

from ._session_tool_test_helpers import (
    _assert_tool_turn_recovers_without_replaying_effect,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "sql", "split_sqlite"))
@pytest.mark.parametrize("phase", ("before_terminal", "after_terminal"))
async def test_session_tool_turn_recovers_at_terminal_commit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(tmp_path, backend, phase)
