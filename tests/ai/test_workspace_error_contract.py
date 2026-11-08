#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration regressions for native workspace error contracts."""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, Literal, cast

import pytest
from pydantic_ai.workspaces import Workspace

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceOperationRejected
from linktools.ai.workspace._bubblewrap import _read_workspace_bytes
from linktools.ai.workspace._sandbox_protocol import validate_request_params

pytestmark = pytest.mark.asyncio


async def _native_file_operation(
    workspace: Workspace,
    operation: Literal["read", "write", "stat", "list", "mkdir", "remove"],
    path: str,
) -> None:
    if operation == "read":
        await workspace.read_bytes(path)
    elif operation == "write":
        await workspace.write_bytes(path, b"replacement")
    elif operation == "stat":
        await workspace.stat(path)
    elif operation == "list":
        await workspace.list_dir(path)
    elif operation == "mkdir":
        await workspace.make_dir(path)
    else:
        await workspace.remove(path)


@pytest.mark.parametrize(
    ("operation", "path", "error_type"),
    [
        ("read", "directory", IsADirectoryError),
        ("write", "directory", IsADirectoryError),
        ("list", "file", NotADirectoryError),
        ("read", "missing", FileNotFoundError),
        ("stat", "missing", FileNotFoundError),
        ("list", "missing", FileNotFoundError),
        ("remove", "missing", FileNotFoundError),
        ("mkdir", "file", FileExistsError),
        ("write", "file/child", NotADirectoryError),
        ("mkdir", "file/child", NotADirectoryError),
    ],
)
async def test_native_filesystem_reports_builtin_path_errors(
    tmp_path: Path,
    operation: Literal["read", "write", "stat", "list", "mkdir", "remove"],
    path: str,
    error_type: type[OSError],
) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "file").write_bytes(b"original")
    session = LocalWorkspaceBackend(tmp_path)
    try:
        with pytest.raises(error_type):
            await _native_file_operation(Workspace(session), operation, path)
        assert (tmp_path / "directory").is_dir()
        assert (tmp_path / "file").read_bytes() == b"original"
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("phase", "error_type", "code"),
    [
        ("write", FileNotFoundError, ErrorCode.STORAGE_NOT_FOUND),
        ("write", IsADirectoryError, ErrorCode.REQUEST_FIELD_INVALID),
        ("write", NotADirectoryError, ErrorCode.REQUEST_FIELD_INVALID),
        ("write", FileExistsError, ErrorCode.STORAGE_CONFLICT),
        ("write", PermissionError, ErrorCode.AUTHORIZATION_DENIED),
        ("before_write", PermissionError, ErrorCode.AUTHORIZATION_DENIED),
        ("flush", PermissionError, ErrorCode.AUTHORIZATION_DENIED),
        ("close", PermissionError, ErrorCode.AUTHORIZATION_DENIED),
    ],
)
async def test_native_mutation_failure_does_not_claim_no_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    error_type: type[OSError],
    code: ErrorCode,
) -> None:
    target = tmp_path / "file"
    target.write_bytes(b"original")
    path_open = Path.open
    failed = False

    def open_with_failure(path: Path, mode: str = "r") -> BinaryIO:
        stream = cast(BinaryIO, path_open(path, mode))
        if path != target or mode != "wb":
            return stream
        write, flush, close = stream.write, stream.flush, stream.close

        def fail() -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise error_type("failure after opening")

        def failing_write(data: bytes) -> int:
            if phase == "before_write":
                fail()
            result = write(data)
            flush()
            fail()
            return result

        def failing_flush() -> None:
            flush()
            fail()

        def failing_close() -> None:
            close()
            fail()

        if phase in {"write", "before_write"}:
            monkeypatch.setattr(stream, "write", failing_write)
        elif phase == "flush":
            monkeypatch.setattr(stream, "flush", failing_flush)
        else:
            monkeypatch.setattr(stream, "close", failing_close)
        return stream

    session = LocalWorkspaceBackend(tmp_path)
    try:
        await session.working_dir()
        monkeypatch.setattr(Path, "open", open_with_failure)
        with pytest.raises(AIError) as raised:
            await Workspace(session).write_bytes("file", b"changed")
        assert raised.value.code is code
        assert not isinstance(raised.value, WorkspaceOperationRejected)
        assert failed
        assert target.read_bytes() == (b"" if phase == "before_write" else b"changed")
    finally:
        await session.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file permissions")
@pytest.mark.parametrize("operation", ["read", "write"])
async def test_native_filesystem_permission_refusals_use_builtin_errors(
    tmp_path: Path, operation: Literal["read", "write"],
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses file permission refusals")
    target = tmp_path / "file"
    target.write_bytes(b"original")
    session = LocalWorkspaceBackend(tmp_path)
    target.chmod(0)
    try:
        with pytest.raises(PermissionError):
            await _native_file_operation(Workspace(session), operation, "file")
    finally:
        target.chmod(0o600)
        assert target.read_bytes() == b"original"
        await session.close()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory permissions")
@pytest.mark.parametrize("operation", ["mkdir", "remove"])
async def test_native_directory_entry_permission_refusals_use_builtin_errors(
    tmp_path: Path, operation: Literal["mkdir", "remove"],
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permission refusals")
    parent = tmp_path / "locked"
    parent.mkdir()
    (parent / "file").write_bytes(b"original")
    session = LocalWorkspaceBackend(tmp_path)
    parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            await _native_file_operation(
                Workspace(session), operation, "locked/new" if operation == "mkdir" else "locked/file",
            )
    finally:
        parent.chmod(0o700)
        assert (parent / "file").read_bytes() == b"original"
        assert not (parent / "new").exists()
        await session.close()


@pytest.mark.parametrize("operation", ["write", "mkdir"])
async def test_native_filesystem_creates_missing_parent_directories(
    tmp_path: Path, operation: Literal["write", "mkdir"],
) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    try:
        await _native_file_operation(Workspace(session), operation, "parent/nested/target")
        target = tmp_path / "parent/nested/target"
        if operation == "write":
            assert target.read_bytes() == b"replacement"
        else:
            assert target.is_dir()
    finally:
        await session.close()


async def test_native_open_failure_after_parent_creation_keeps_unknown_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "parent/nested/file"
    path_open = Path.open

    def denied_open(path: Path, mode: str = "r") -> BinaryIO:
        if path == target and mode == "wb":
            raise PermissionError("opening refused after parent creation")
        return cast(BinaryIO, path_open(path, mode))

    session = LocalWorkspaceBackend(tmp_path)
    try:
        await session.working_dir()
        monkeypatch.setattr(Path, "open", denied_open)
        with pytest.raises(AIError) as raised:
            await Workspace(session).write_bytes("parent/nested/file", b"changed")
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not isinstance(raised.value, WorkspaceOperationRejected)
        assert target.parent.is_dir()
        assert not target.exists()
    finally:
        await session.close()


@pytest.mark.parametrize("operation", ["mkdir", "remove"])
async def test_native_partial_directory_mutation_keeps_unknown_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: Literal["mkdir", "remove"],
) -> None:
    directory = tmp_path / "directory"
    path_mkdir = Path.mkdir

    def partial_mkdir(path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if path == directory / "nested/target" and parents:
            path_mkdir(directory)
            raise PermissionError("creation refused after creating one parent")
        path_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def partial_remove(path: Path) -> None:
        (path / "first").unlink()
        raise PermissionError("removal refused after unlinking one child")

    session = LocalWorkspaceBackend(tmp_path)
    try:
        await session.working_dir()
        if operation == "mkdir":
            monkeypatch.setattr(Path, "mkdir", partial_mkdir)
            path = "directory/nested/target"
        else:
            directory.mkdir()
            (directory / "first").write_bytes(b"first")
            (directory / "second").write_bytes(b"second")
            monkeypatch.setattr(shutil, "rmtree", partial_remove)
            path = "directory"
        with pytest.raises(AIError) as raised:
            await _native_file_operation(Workspace(session), operation, path)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not isinstance(raised.value, WorkspaceOperationRejected)
        assert directory.is_dir()
        if operation == "mkdir":
            assert not (directory / "nested").exists()
        else:
            assert not (directory / "first").exists()
            assert (directory / "second").read_bytes() == b"second"
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (ErrorCode.REQUEST_FIELD_INVALID, None),
        (ErrorCode.REQUEST_FIELD_INVALID, "file_byte_limit"),
        (ErrorCode.STORAGE_CONFLICT, "file_changed"),
        (ErrorCode.STORAGE_UNAVAILABLE, "is_directory"),
    ],
)
async def test_native_filesystem_preserves_unclassified_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: ErrorCode,
    reason: str | None,
) -> None:
    error = AIError(code, retryable=True, safe_details={} if reason is None else {"reason": reason})

    async def fail(
        operation: str, path: str, *, data: bytes | None = None,
    ) -> Mapping[str, JsonValue]:
        raise error

    session = LocalWorkspaceBackend(tmp_path)
    monkeypatch.setattr(session, "raw_filesystem", fail)
    try:
        with pytest.raises(AIError) as raised:
            await Workspace(session).stat("file")
        assert raised.value is error
        assert raised.value.retryable
    finally:
        await session.close()


async def test_logical_file_reads_keep_structured_errors(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    session = LocalWorkspaceBackend(tmp_path)
    try:
        with pytest.raises(AIError) as raised:
            await session.read_bytes("directory")
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await session.close()


@pytest.mark.parametrize("path", [".", "directory"])
async def test_bubblewrap_reader_classifies_directory_failures(tmp_path: Path, path: str) -> None:
    (tmp_path / "directory").mkdir()
    with pytest.raises(AIError) as raised:
        _read_workspace_bytes(tmp_path, path, read_policy=None, hidden_paths=(), max_bytes=None)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details["reason"] == "is_directory"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
async def test_native_read_refuses_special_files_without_waiting(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")
    session = LocalWorkspaceBackend(tmp_path)
    try:
        with pytest.raises(OSError, match="not a regular file"):
            await Workspace(session).read_bytes("pipe")
        with pytest.raises(AIError) as raised:
            _read_workspace_bytes(tmp_path, "pipe", read_policy=None, hidden_paths=(), max_bytes=None)
        assert raised.value.safe_details["reason"] == "not_regular_file"
    finally:
        await session.close()


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
