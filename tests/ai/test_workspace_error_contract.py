#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration regressions for native workspace error contracts."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalWorkspaceBackend
from linktools.ai.workspace._sandbox_protocol import validate_request_params

pytestmark = pytest.mark.asyncio


async def test_sandbox_protocol_rejects_unknown_or_missing_fields() -> None:
    invalid_requests = (
        ("read_text", {}),
        ("read_text", {"path": "a", "unknown": None}),
        ("unknown", {}),
    )
    for method, params in invalid_requests:
        with pytest.raises(AIError) as raised:
            validate_request_params(method, params)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_missing_file_is_reported_as_not_found(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(AIError) as raised:
            await session.read_text("missing.json")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND


async def test_write_creates_missing_parent_directories(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        await session.write_file("worker/personnel_context/report.md", "report")
        assert (tmp_path / "worker/personnel_context/report.md").read_text() == "report"
    finally:
        await session.close()
