#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox lifecycle boundary regressions."""

import asyncio
import sys
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from linktools.ai.capability import ShellCapability
from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._compaction import (
    CompactionCapability,
    CompactionPolicy,
)
from linktools.ai.workspace import WorkspaceOperationRejected, WorkspaceResource, Workspace
from linktools.ai.workspace import BubblewrapWorkspaceBackend, WorkspaceReadPolicy
from linktools.ai.workspace import _bubblewrap
from linktools.ai.workspace._bubblewrap import (
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
from pydantic_ai.capabilities import AbstractCapability


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
    )


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Bubblewrap requires Linux")
@pytest.mark.parametrize("hidden_paths", ((), ("private/nested",)))
async def test_read_only_sandbox_cleans_locks_after_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_paths: tuple[str, ...],
) -> None:
    root = tmp_path / "workspace"
    runtime_root = tmp_path / "rootfs"
    root.mkdir()
    (root / ".linktools").mkdir()
    runtime_root.mkdir()
    lock_root = tmp_path / "locks"

    def create_locks(*, prefix: str) -> str:
        lock_root.mkdir()
        return str(lock_root)

    def reject_rootfs(*args: object) -> None:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)

    monkeypatch.setattr(_bubblewrap.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(_bubblewrap.os, "pidfd_open", lambda pid: -1, raising=False)
    monkeypatch.setattr(_bubblewrap.tempfile, "mkdtemp", create_locks)
    monkeypatch.setattr(_bubblewrap, "_validate_rootfs", reject_rootfs)
    sandbox = partial(BubblewrapWorkspaceBackend, runtime_root=runtime_root, bwrap_executable=Path(sys.executable).resolve(), hidden_paths=hidden_paths, read_policy=WorkspaceReadPolicy(("**",)))
    with pytest.raises(AIError) as raised:
        await sandbox(root=root).working_dir()
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert not lock_root.exists()
    assert tuple(root.iterdir()) == (root / ".linktools",)


def test_read_only_bubblewrap_mounts_workspace_read_only(tmp_path: Path) -> None:
    arguments = _build_bwrap_args(
        root=tmp_path, runtime_root=tmp_path / "runtime",
        bwrap=Path("/usr/bin/bwrap"), lock_root=tmp_path / "locks",
        resources=(), hidden_paths=(), worker_resources=[],
        read_policy=WorkspaceReadPolicy(("**",)),
    )
    workspace_mount = arguments.index("/workspace")
    assert arguments[workspace_mount - 2] == "--ro-bind"


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run_command(self, command: str, *, timeout_seconds=None):
        self.calls.append(command)
        return "ok"

    async def start_command(self, command: str):
        raise NotImplementedError

    async def check_command(self, command_id: str):
        raise NotImplementedError

    async def stop_command(self, command_id: str):
        raise NotImplementedError


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
    toolset = ShellCapability().bind(("run_command",), backend=session).get_toolset()
    context = _context()
    tools = await toolset.get_tools(context)
    result = await toolset.call_tool("run_command", {"command": "echo ok"}, context, tools["run_command"])

    assert "ok" in result
    assert len(session.calls) == 1
    assert session.calls[0] == "echo ok"


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
        (ERROR_EFFECT_NOT_APPLIED, WorkspaceOperationRejected),
        (ERROR_EFFECT_UNKNOWN, AIError),
    ),
)
@pytest.mark.asyncio
async def test_bubblewrap_error_frame_preserves_effect_certainty(
    effect: str,
    expected_type: type[AIError],
    tmp_path: Path,
) -> None:
    session = BubblewrapWorkspaceBackend(tmp_path, runtime_root=tmp_path,
                                        bwrap_executable=Path("/usr/bin/bwrap"))
    session._pending = {}
    session._pending_lock = asyncio.Lock()
    session._structured_requests = set()
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


def test_runtime_compaction_uses_current_file_deduplication() -> None:
    compaction = CompactionCapability(
        4096,
        limits=PromptLimits(),
        policy=CompactionPolicy(
            context_dedupe_by_tool={
                "read_file": "workspace_file_read_v1",
            },
        ),
        journal=None,
        observer=None,
        projection_sink=None,
    )

    assert isinstance(compaction._deduplicate, AbstractCapability)
    assert compaction._target_tokens == 4096


def test_bubblewrap_hidden_mount_overrides_workspace_resource_bind(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "package"
    source.mkdir()
    (source / "private").mkdir()
    resource = WorkspaceResource("resource", source)

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
    resource = WorkspaceResource("resource", source)

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
