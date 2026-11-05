#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session tool result recovery around terminal commit process exits."""

from pathlib import Path

import pytest

from ._session_tool_test_helpers import (
    _assert_tool_turn_recovers_without_replaying_effect,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", (
    "sqlite",
    pytest.param("sql", marks=pytest.mark.merge),
    pytest.param("split_sqlite", marks=pytest.mark.merge),
))
@pytest.mark.parametrize("phase", ("before_terminal", "after_terminal"))
async def test_session_tool_turn_recovers_at_terminal_commit_without_replaying_effect(
    tmp_path: Path,
    backend: str,
    phase: str,
) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(tmp_path, backend, phase)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing_source", "missing_latest", "interrupted_latest"))
async def test_completed_native_resume_does_not_hide_missing_or_partial_trace(tmp_path: Path, damage: str) -> None:
    await _assert_tool_turn_recovers_without_replaying_effect(
        tmp_path, "sqlite", "before_terminal", trace_damage=damage,
    )
