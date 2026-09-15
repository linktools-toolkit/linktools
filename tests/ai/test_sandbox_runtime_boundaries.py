#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox lifecycle boundary regressions."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from linktools.ai.capability import workspace_capabilities
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._compaction import (
    RuntimeCompaction,
    RuntimeCompactionPolicy,
)
from linktools.ai.workspace import SandboxOperationRejected, SandboxResource, Workspace
from linktools.ai.workspace._bubblewrap import (
    _BubblewrapSandboxSession,
    _build_bwrap_args,
)
from linktools.ai.workspace._sandbox_protocol import (
    ERROR_EFFECT_NOT_APPLIED,
    ERROR_EFFECT_UNKNOWN,
    MAX_SAFE_DETAILS_BYTES,
    read_frame,
)
from linktools.ai.workspace.sandbox_worker import _send_error
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import DeduplicateFileReads


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        del timeout_seconds
        self.calls.append(command)
        return "ok"


class _BufferWriter:
    def __init__(self) -> None:
        self.payload = bytearray()

    def write(self, value: bytes) -> None:
        self.payload.extend(value)

    async def drain(self) -> None:
        pass


@pytest.mark.asyncio
async def test_workspace_capability_adapts_the_caller_owned_session(
    tmp_path: Path,
) -> None:
    session = _FakeSession()
    capability = workspace_capabilities(
        Workspace.load(tmp_path, workspace_id="workspace"),
        ("run_command",),
        session=session,  # type: ignore[arg-type]
    )[0]
    toolset = capability.get_toolset()
    tools = await toolset.get_tools(_context())
    result = await toolset.call_tool(
        "run_command",
        {"command": "echo ok"},
        _context(),
        tools["run_command"],
    )

    assert result == "ok"
    assert session.calls == ["echo ok"]


class _CancellingCloseSession:
    async def close(self) -> None:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_cancelled_close_does_not_replace_primary_failure(tmp_path: Path) -> None:
    del tmp_path
    session = _CancellingCloseSession()
    with pytest.raises(asyncio.CancelledError):
        await session.close()


@pytest.mark.asyncio
async def test_cancelled_close_propagates_without_primary_failure(tmp_path: Path) -> None:
    del tmp_path
    session = _CancellingCloseSession()
    with pytest.raises(asyncio.CancelledError):
        await session.close()


@pytest.mark.parametrize(
    ("effect", "expected_type"),
    (
        (ERROR_EFFECT_NOT_APPLIED, SandboxOperationRejected),
        (ERROR_EFFECT_UNKNOWN, AIError),
    ),
)
@pytest.mark.asyncio
async def test_bubblewrap_error_frame_preserves_effect_certainty(
    effect: str,
    expected_type: type[AIError],
) -> None:
    session = _BubblewrapSandboxSession(cast(Any, object()), {})
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    session._pending["request"] = (future, True)

    await session._accept_response(
        {
            "request_id": "request",
            "error": {
                "code": ErrorCode.REQUEST_FIELD_INVALID.value,
                "safe_details": {"reason": "invalid"},
                "effect": effect,
            },
        }
    )

    error = future.exception()
    assert type(error) is expected_type
    assert isinstance(error, AIError)
    assert error.code is ErrorCode.REQUEST_FIELD_INVALID
    assert error.safe_details == {"reason": "invalid"}


@pytest.mark.asyncio
async def test_sandbox_worker_bounds_safe_details_independently_from_frame() -> None:
    payload_overhead = len(b'{"reason":""}')
    details = {"reason": "x" * (MAX_SAFE_DETAILS_BYTES - payload_overhead)}
    writer = _BufferWriter()

    await _send_error(
        cast(asyncio.StreamWriter, writer),
        "request",
        ErrorCode.REQUEST_FIELD_INVALID,
        asyncio.Lock(),
        details,
        effect=ERROR_EFFECT_NOT_APPLIED,
    )

    assert len(writer.payload) > MAX_SAFE_DETAILS_BYTES
    reader = asyncio.StreamReader()
    reader.feed_data(bytes(writer.payload))
    reader.feed_eof()
    frame = await read_frame(reader)
    assert frame is not None
    assert frame["error"]["safe_details"] == details


def test_runtime_compaction_uses_harness_deduplication() -> None:
    compaction = RuntimeCompaction(
        4096,
        policy=RuntimeCompactionPolicy(
            context_dedupe_by_tool={
                "read_file": "workspace_file_read_v1",
            },
        ),
        journal=None,
        observer=None,
        projection_sink=None,
    )

    assert isinstance(compaction._deduplicate, DeduplicateFileReads)
    assert compaction._target_tokens == 4096


def test_bubblewrap_hidden_mount_overrides_workspace_resource_bind(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "package"
    source.mkdir()
    (source / "private").mkdir()
    resource = SandboxResource("resource", source)

    arguments = _build_bwrap_args(
        root=workspace,
        runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"),
        lock_root=workspace / ".linktools" / "locks",
        resources=(resource,),
        hidden_paths=("package/private",),
        worker_resources=[{"key": "resource", "path": "/skills/resource"}],
    )

    visible_bind = arguments.index("/workspace/package")
    hidden_mount = arguments.index("/workspace/package/private")
    assert hidden_mount > visible_bind


def test_bubblewrap_allows_explicit_skill_resource_under_hidden_storage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / ".linktools" / "skills" / "review"
    source.mkdir(parents=True)
    resource = SandboxResource("resource", source)

    arguments = _build_bwrap_args(
        root=workspace,
        runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"),
        lock_root=workspace / ".linktools" / "locks",
        resources=(resource,),
        hidden_paths=(".linktools",),
        worker_resources=[{"key": "resource", "path": "/skills/resource"}],
    )

    assert arguments.count(str(source)) == 1
    assert "/workspace/.linktools/skills/review" not in arguments
