#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration regressions for native workspace error contracts."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalSandbox
from linktools.ai.workspace._sandbox_protocol import validate_request_params

pytestmark = pytest.mark.asyncio


async def test_sandbox_protocol_rejects_unknown_or_missing_fields() -> None:
    invalid_requests = (
        ("read_file", {}),
        ("read_file", {"path": "a", "unknown": None}),
        ("unknown", {}),
    )
    for method, params in invalid_requests:
        with pytest.raises(AIError) as raised:
            validate_request_params(method, params)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_missing_file_is_reported_as_not_found(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    try:
        with pytest.raises(AIError) as raised:
            await session.read_file("missing.json")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND


async def test_missing_write_parent_is_reported_as_not_found(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    try:
        with pytest.raises(AIError) as raised:
            await session.write_file(
                "worker/personnel_context/report.md",
                "report",
            )
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
