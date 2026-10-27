#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution backend.

The local backend provides the sandbox contract and lifecycle guarantees; it
is deliberately not a security sandbox.  Agent code still only receives the
logical file and command operations exposed by the native workspace backend.
"""

import asyncio
import shlex
import shutil
import codecs
import ctypes
import fnmatch
import hashlib
import mimetypes
import os
import re
import stat
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import DEVNULL, PIPE
from typing import BinaryIO, TypeVar

from filelock import FileLock
from linktools.core import environ

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._native import _NativeWorkspaceOperations
from ._sandbox import (
    WorkspaceOperationRejected,
    WorkspaceReadPolicy,
    WorkspaceResource,
    WorkspaceResourcePath,
    WorkspaceStdioProcess,
    _normalize_stdio_environment,
    _stdio_resource_root,
    normalize_workspace_input_path,
)
from ._root import Workspace
from ._file_io import (
    _check_state, _facts, _glob_matches, _inspect_file, _read_text_range, _state,
)
from ._sandbox_protocol import validate_request_size
from ._local_process import (
    _ProcessState,
    _WindowsJob,
    _command_result,
    _finish_task,
    _stop_process,
    _stop_process_state,
    _terminate_unregistered_process,
)

_logger = environ.get_logger("ai.workspace.local_sandbox")
_MAX_RENDERED_CHARS = 50_000
_MAX_CONTENT_CHARS = 50_000
_MAX_ITEMS = 1_000
_MAX_SEARCH_PATTERN_CHARS = 1_000
_MAX_WRITE_CHARS = 10 * 1024 * 1024
_MAX_EDIT_BYTES = 256 * 1024 * 1024
_MAX_FILE_SECONDS = 5.0
_MAX_ENUMERATED_ITEMS = 10_000
_SyncResult = TypeVar("_SyncResult")
_DEFAULT_COMMAND_TIMEOUT = 30.0
_PROTECTED_NAMES = (".env", ".env.*", "*.pem", "*.key")
_SUDO_COMMAND = re.compile(r"^sudo\s")
_INTERACTIVE_COMMANDS = re.compile(
    r"^(vi|vim|nano|emacs|less|more|top|htop|man)\b"
)
_REMOTE_COMMANDS = re.compile(r"^(passwd|ssh|telnet|ftp)\b")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


_DANGEROUS_COMMANDS = frozenset(
    {
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "format",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
    }
)


class LocalWorkspaceBackend(_NativeWorkspaceOperations):
    def __init__(
        self,
        root: Path,
        *,
        resources: tuple[WorkspaceResource, ...] = (),
        read_policy: WorkspaceReadPolicy | None = None,
        allow_host_stdio_with_read_policy: bool = False,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a Path")
        if read_policy is not None and not isinstance(read_policy, WorkspaceReadPolicy):
            raise TypeError("read_policy must be WorkspaceReadPolicy")
        if not isinstance(allow_host_stdio_with_read_policy, bool):
            raise TypeError("allow_host_stdio_with_read_policy must be bool")
        self._root = root.absolute()
        self._ref_id: str | None = None
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self._resource_config = resources
        self._workspace: Workspace
        self._resources = {
            resource.id: (resource.source if resource.source is not None else None, resource)
            for resource in resources
        }
        self._lock_root: Path | None = None
        self._read_policy = read_policy
        self._allow_host_stdio_with_read_policy = allow_host_stdio_with_read_policy
        self._environment = _command_environment()
        self._state = "NEW"
        self._state_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._processes: dict[str, _ProcessState] = {}
        self._stdio_processes: set[_LocalStdioProcess] = set()
        self._pending_processes: dict[
            int, tuple[asyncio.subprocess.Process, _WindowsJob | None]
        ] = {}
        self._operations: set[asyncio.Task[object]] = set()
        self._starting_operations: set[asyncio.Future[None]] = set()
        self._starting_background = 0
        self._process_lock = asyncio.Lock()

    async def _initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            if self._state != "NEW":
                raise AIError(_session_state_error(self._state))
            self._ref_id = uuid.uuid4().hex
            try:
                self._root = _normalize_root(self._root)
                self._workspace = Workspace(self._root, {})
                resources = _validate_resources(self._workspace, self._resource_config)
                self._resources = {
                    resource.id: (resource.source.resolve() if resource.source is not None else None, resource)
                    for resource in resources
                }
                if self._lock_root is None:
                    self._lock_root = self._workspace.locks_root
                self._state = "OPEN"
                self._initialized = True
            except BaseException:
                self._state = "LOST"
                raise

    def stdio_execution_policy(self) -> Mapping[str, JsonValue]:
        if self._read_policy is not None and not self._allow_host_stdio_with_read_policy:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE,
                          safe_details={"reason": "local_stdio_read_policy_unsupported"})
        return {"version": 1, "boundary": "host-stdio"}

    @property
    def execution_root(self) -> str:
        return self._root.as_posix()

    async def raw_filesystem(
        self, operation: str, path: str, *, data: bytes | None = None,
    ) -> Mapping[str, JsonValue]:
        """Execute raw file primitives under the same path and resource authority."""
        def perform() -> Mapping[str, JsonValue]:
            selected_path = path
            if operation == "realpath":
                try:
                    selected_path = (self._root / path).resolve(strict=False).as_posix()
                except (OSError, RuntimeError) as error:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error
            mutation = operation in {"write", "mkdir", "remove"}
            if mutation:
                self._reject_read_only()
                normalized = _normalize_path(path)
                if _is_protected(self._workspace, normalized):
                    raise WorkspaceOperationRejected(ErrorCode.AUTHORIZATION_DENIED)
                target = self._resolve_path(normalized, allow_missing=True)
                if _is_protected(self._workspace, _relative(self._root, target.resolve())):
                    raise WorkspaceOperationRejected(ErrorCode.AUTHORIZATION_DENIED)
            else:
                target, normalized, resource = self._read_target(
                    selected_path, allow_missing=True, directory=None,
                )
            if operation == "realpath":
                return {"path": target.resolve(strict=False).as_posix()}
            if operation == "exists":
                return {"exists": target.exists()}
            if operation == "write":
                if not isinstance(data, bytes):
                    raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID)
                if target.exists() and not target.is_file():
                    raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return {}
            if operation == "mkdir":
                target.mkdir(parents=True, exist_ok=True)
                return {}
            if operation == "remove":
                if target == self._root or _inside_or_equal(target, self._root):
                    raise WorkspaceOperationRejected(ErrorCode.AUTHORIZATION_DENIED)
                original = self._root / normalized
                if original.is_symlink():
                    original.unlink()
                elif target.is_dir():
                    for child in target.rglob("*"):
                        if _is_protected(self._workspace, _relative(self._root, child)):
                            raise WorkspaceOperationRejected(ErrorCode.AUTHORIZATION_DENIED)
                    shutil.rmtree(target)
                else:
                    target.unlink()
                return {}
            if operation == "stat":
                return self._raw_entry(target)
            if operation == "list":
                items = []
                for child in target.iterdir():
                    relative = (PurePosixPath(normalized) / child.name).as_posix()
                    if self._visible_read_path(relative, resource_key=resource, directory=child.is_dir()):
                        # Resolve each child before exposing metadata through a symlink.
                        self._read_target(child.as_posix(), directory=None)
                        items.append(self._raw_entry(child))
                return {"entries": items}
            raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID)
        try:
            return await self._run_sync(perform)
        except FileNotFoundError as error:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
        except (IsADirectoryError, NotADirectoryError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    def _raw_entry(self, path: Path) -> dict[str, JsonValue]:
        info = path.stat()
        return {"name": path.name, "path": path.as_posix(),
                "is_dir": stat.S_ISDIR(info.st_mode), "size": info.st_size}

    async def execute_command(
        self, command: str, *, environment: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, JsonValue]:
        """Execute an authorized backend command and preserve structured output."""
        self._reject_read_only()
        try:
            environment = _normalize_stdio_environment(environment)
            _validate_command_input(command)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error
        await self._ensure_open()
        process = await self._start_process(command, environment=environment)
        timed_out = False
        cleanup_succeeded = True
        try:
            await asyncio.wait_for(asyncio.shield(process.wait_task), timeout_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            cleanup_succeeded = False
            await _stop_process(process, force=True)
            cleanup_succeeded = True
        except BaseException:
            try:
                await _finish_task(asyncio.create_task(_stop_process(process, force=True)))
            except BaseException:
                cleanup_succeeded = False
                raise
            raise
        finally:
            if cleanup_succeeded:
                await self._forget_process(process.command_id)
        return {"exit_code": process.process.returncode, "stdout": "".join(process.stdout),
                "stderr": "".join(process.stderr), "timed_out": timed_out,
                "output_truncated": process.output_truncated}

    def resource_path(self, resource_id: str) -> "str | None":
        self._ensure_open_sync()
        if not isinstance(resource_id, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if resource_id not in self._resources:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source, _resource = self._resources[resource_id]
        if source is None:
            return None
        if not self._visible_read_path(".", resource_key=resource_id, directory=True):
            return None
        return str(source)

    def managed_process_ids(self) -> frozenset[int]:
        """Return child IDs still owned by this worker session."""
        values = {
            process.process.pid
            for process in self._processes.values()
            if (
                process.process.pid is not None
                and not process.wait_task.done()
            )
        }
        values.update(
            process.pid
            for process, _job in self._pending_processes.values()
            if process.pid is not None
        )
        values.update(
            process._process.pid
            for process in self._stdio_processes
            if process._process.pid is not None
            and process._process.returncode is None
        )
        return frozenset(values)

    async def open_stdio_process(
        self,
        command: str,
        args: "Sequence[str | WorkspaceResourcePath]" = (),
        *,
        resources: "Sequence[WorkspaceResource]" = (),
        environment: "Mapping[str, str] | None" = None,
        cwd_resource_id: str | None = None,
    ) -> WorkspaceStdioProcess:
        """Open a trusted host process outside the file API's read policy."""
        if self._read_policy is not None and not self._allow_host_stdio_with_read_policy:
            raise AIError(
                ErrorCode.SANDBOX_UNAVAILABLE,
                safe_details={"reason": "local_stdio_read_policy_unsupported"},
            )
        if (
            not isinstance(command, str)
            or not command
            or "\x00" in command
            or isinstance(args, (str, bytes, bytearray))
            or isinstance(resources, (str, bytes, bytearray))
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        if os.name != "posix":
            raise WorkspaceOperationRejected(ErrorCode.SANDBOX_UNAVAILABLE,
                                              safe_details={"reason": "posix_execution_required"})
        await self._ensure_open()
        selected_resources = _validate_resources(
            self._workspace,
            tuple(resources),
        )
        resource_root = _stdio_resource_root(cwd_resource_id, selected_resources)
        command_args = _local_stdio_command_args(args, selected_resources)
        process_environment = dict(self._environment)
        process_environment.update(_normalize_stdio_environment(environment))
        if resource_root is not None:
            process_environment["PWD"] = str(resource_root)
        started = asyncio.get_running_loop().create_future()
        async with self._process_lock:
            if self._state != "OPEN":
                raise AIError(_session_state_error(self._state))
            self._starting_operations.add(started)

        job: _WindowsJob | None = None
        process: asyncio.subprocess.Process | None = None
        process_key: int | None = None
        try:
            if os.name == "nt":
                job = _WindowsJob.create()
            process_kwargs: dict[str, object] = {
                "cwd": str(self._root if resource_root is None else resource_root),
                "env": process_environment,
                "stdin": PIPE,
                "stdout": PIPE,
                "stderr": PIPE,
            }
            if os.name == "nt":
                process_kwargs["creationflags"] = 0x00000004
            else:
                process_kwargs["start_new_session"] = True

            process_task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    command,
                    *command_args,
                    **process_kwargs,
                ),
                name="local-sandbox-stdio-start",
            )
            try:
                process = await asyncio.shield(process_task)
            except asyncio.CancelledError as cancellation:
                try:
                    process = await _finish_task(process_task)
                except BaseException:
                    if job is not None:
                        job.close()
                    raise
                process_key = id(process)
                self._pending_processes[process_key] = (process, job)
                try:
                    await _finish_task(asyncio.create_task(
                        _terminate_unregistered_process(process, job)
                    ))
                except BaseException as cleanup_error:
                    if process.returncode is not None:
                        self._pending_processes.pop(process_key, None)
                    raise cancellation from cleanup_error
                self._pending_processes.pop(process_key, None)
                raise cancellation
            except (OSError, ValueError) as error:
                if job is not None:
                    job.close()
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error

            process_key = id(process)
            self._pending_processes[process_key] = (process, job)
            if job is not None:
                try:
                    job.assign_and_resume(process.pid)
                except BaseException as error:
                    try:
                        await _terminate_unregistered_process(process, job)
                    finally:
                        self._pending_processes.pop(process_key, None)
                    if isinstance(error, AIError):
                        raise
                    raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error

            stdio_process = _LocalStdioProcess(
                process,
                job,
                self._stdio_processes.discard,
            )
            async with self._process_lock:
                if self._state != "OPEN":
                    await stdio_process.close()
                    self._pending_processes.pop(process_key, None)
                    raise AIError(_session_state_error(self._state))
                self._stdio_processes.add(stdio_process)
                self._pending_processes.pop(process_key, None)
            return stdio_process
        finally:
            started.set_result(None)
            self._starting_operations.discard(started)

    async def canonicalize_path(self, path: str) -> str:
        """Normalize one logical Workspace path without applying operation policy."""
        def operation() -> str:
            return _normalize_path(path)

        return await self._run_sync(operation)

    async def _read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
        expected_state: str | None = None,
    ) -> bytes:
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if expected_state is not None and not isinstance(expected_state, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        def operation() -> bytes:
            target, display, _resource_key = self._read_target(path)
            try:
                if not stat.S_ISREG(target.stat().st_mode):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                with target.open("rb") as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    state = _state(display, info)
                    if expected_state is not None and expected_state != state:
                        raise AIError(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "file_changed"})
                    if max_bytes is not None and info.st_size > max_bytes:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "file_byte_limit"})
                    value = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
                    _check_state(target, display, stream, state)
                if max_bytes is not None and len(value) > max_bytes:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "file_byte_limit"})
                return value
            except AIError:
                raise
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

        return await self._run_sync(operation)

    async def inspect_file(
        self, path: str, *, max_bytes: int | None = None,
    ) -> Mapping[str, JsonValue]:
        """Inspect only an authorized, bounded header; never expand containers."""
        validate_request_size("inspect_file", {"path": path, "max_bytes": max_bytes})
        def operation() -> Mapping[str, JsonValue]:
            target, display, _resource_key = self._read_target(path)
            try:
                return _inspect_file(target, display, max_bytes)
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        return await self._run_sync(operation)

    async def read_text(
        self, path: str, *, offset: int | None = None, limit: int | None = None,
        cursor: str | None = None, full_metadata: bool = False,
        max_bytes: int | None = None, max_read_bytes: int | None = None,
    ) -> Mapping[str, JsonValue]:
        """Read a bounded UTF-8 preview without implicitly scanning the file."""
        validate_request_size("read_text", {
            "path": path, "offset": offset, "limit": limit, "cursor": cursor,
            "full_metadata": full_metadata, "max_bytes": max_bytes,
            "max_read_bytes": max_read_bytes,
        })
        def operation() -> Mapping[str, JsonValue]:
            target, display, _resource_key = self._read_target(path)
            try:
                return _read_text_range(
                    target, display, offset=offset, limit=limit,
                    cursor=cursor, full_metadata=full_metadata, max_bytes=max_bytes,
                    max_read_bytes=max_read_bytes,
                )
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        return await self._run_sync(operation)


    async def write_file(
        self,
        path: str,
        content: str,
        *,
        old_text: str | None = None,
        expected_hash: str | None = None,
    ) -> str:
        if old_text is not None:
            return await self._edit_file(path, old_text, content, expected_hash=expected_hash)
        self._reject_read_only()
        try:
            normalized = _normalize_path(path)
            expected = _validate_expected_hash(expected_hash)
            validate_request_size(
                "write_file",
                {"path": path, "content": content, "expected_hash": expected_hash},
            )
            _validate_text(content)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error

        def operation() -> str:
            try:
                target = self._file_path(normalized, write=True, allow_missing_parent=True)
            except AIError as error:
                raise WorkspaceOperationRejected.from_error(error) from error
            with _file_lock(self._lock_root, _relative(self._root, target)):
                if expected is not None:
                    try:
                        current_hash = _read_optional_hash(target, normalized)
                        _check_expected_digest(current_hash, expected)
                    except AIError as error:
                        raise WorkspaceOperationRejected.from_error(error) from error
                _atomic_write(self._root, target, content)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            return _write_result(normalized, digest)

        return await self._run_sync(operation)

    async def _edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        self._reject_read_only()
        try:
            normalized = _normalize_path(path)
            if not isinstance(old_text, str) or not old_text:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            expected = _validate_expected_hash(expected_hash)
            validate_request_size(
                "write_file",
                {
                    "path": path,
                    "old_text": old_text,
                    "content": new_text,
                    "expected_hash": expected_hash,
                },
            )
            _validate_text(old_text)
            _validate_text(new_text)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error

        def operation() -> str:
            try:
                target = self._file_path(normalized, write=True)
            except AIError as error:
                raise WorkspaceOperationRejected.from_error(error) from error
            with _file_lock(self._lock_root, _relative(self._root, target)):
                digest = _stream_edit(
                    self._root, target, normalized, old_text, new_text, expected,
                )
            return _write_result(normalized, digest)

        return await self._run_sync(operation)

    def _enumerate_files(
        self, target: Path, display: str, resource_key: str | None,
        *, max_depth: int | None, state: "_WalkState", deadline: float,
    ) -> Iterable[tuple[Path, str, os.stat_result]]:
        pending = [(target, 0)]
        count = 0
        while pending:
            current, depth = pending.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        count += 1
                        if count > _MAX_ENUMERATED_ITEMS or time.monotonic() > deadline:
                            state.failed = True
                            return
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except OSError:
                            state.failed = True
                            continue
                        is_directory = stat.S_ISDIR(info.st_mode)
                        if not is_directory and not stat.S_ISREG(info.st_mode):
                            continue
                        child = Path(entry.path)
                        logical = _child_display(display, child, target)
                        if not self._visible_read_path(logical, resource_key=resource_key, directory=is_directory):
                            continue
                        yield child, logical, info
                        if is_directory and (max_depth is None or depth + 1 < max_depth):
                            pending.append((child, depth + 1))
            except OSError:
                state.failed = True

    async def list_files(self, path: str = ".", *, include_glob: str = "*") -> str:
        """List names with extension hints and explicit recursive glob traversal."""
        _validate_glob(include_glob)
        validate_request_size("list_files", {"path": path, "include_glob": include_glob})
        def operation() -> str:
            target, display, resource_key = self._read_target(path, directory=True)
            if not target.is_dir():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            walk_state = _WalkState()
            deadline = time.monotonic() + _MAX_FILE_SECONDS
            depth = None if "**" in include_glob.split("/") else len(include_glob.split("/"))
            entries: list[tuple[str, str]] = []
            for candidate, logical, info in self._enumerate_files(
                target, display, resource_key, max_depth=depth,
                state=walk_state, deadline=deadline,
            ):
                relative = candidate.relative_to(target).as_posix()
                if not _glob_matches(relative, include_glob):
                    continue
                if len(entries) == _MAX_ITEMS:
                    walk_state.failed = True
                    break
                kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
                media_type = None if kind == "directory" else mimetypes.guess_type(relative)[0]
                row = f"{logical}  {kind}  {info.st_size}"
                if media_type is not None:
                    row += f"  {media_type}  extension"
                entries.append((logical, row))
            result = "\n".join(row for _logical, row in sorted(entries)) or "(empty directory)"
            if walk_state.failed:
                return _bound_output_with_marker(result, "\n[truncated: narrow path or include_glob]")
            return _bound_output(result)
        return await self._run_sync(operation)

    async def search_text(
        self, pattern: str, *, path: str = ".", include_glob: str | None = None,
    ) -> str:
        """Search bounded text lines using the same path-anchored glob as listing."""
        expression = _compile_pattern(pattern)
        if include_glob is not None:
            _validate_glob(include_glob)
        validate_request_size("search_text", {"pattern": pattern, "path": path, "include_glob": include_glob})
        def operation() -> str:
            target, display, resource_key = self._read_target(path, directory=None)
            walk_state = _WalkState()
            deadline = time.monotonic() + _MAX_FILE_SECONDS
            if target.is_file():
                candidates = iter(((target, display, target.stat()),))
            elif target.is_dir():
                candidates = self._enumerate_files(target, display, resource_key, max_depth=None, state=walk_state, deadline=deadline)
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            rows: list[str] = []
            remaining_scan = 8 * 1024 * 1024
            for candidate, logical, info in candidates:
                if not stat.S_ISREG(info.st_mode):
                    continue
                if not self._visible_read_path(logical, resource_key=resource_key, directory=False):
                    continue
                relative = candidate.name if candidate == target else candidate.relative_to(target).as_posix()
                if include_glob is not None and not _glob_matches(relative, include_glob):
                    continue
                if remaining_scan <= 0 or time.monotonic() > deadline:
                    walk_state.failed = True
                    break
                try:
                    matches, incomplete, scanned = _search_bounded_lines(
                        candidate, logical, expression, remaining_scan, deadline,
                        _MAX_ITEMS - len(rows),
                    )
                except (AIError, OSError):
                    walk_state.failed = True
                    continue
                remaining_scan -= scanned
                walk_state.failed |= incomplete
                rows.extend(matches)
                if len(rows) >= _MAX_ITEMS or sum(map(len, rows)) >= _MAX_RENDERED_CHARS - 64:
                    walk_state.failed = True
                    break
            result = "\n".join(rows) or ("No matches in scanned text." if walk_state.failed else "No matches found.")
            if walk_state.failed:
                return _bound_output_with_marker(result, "\n[truncated: search coverage incomplete]")
            return _bound_output(result)
        return await self._run_sync(operation)


    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        self._reject_read_only()
        try:
            validate_request_size(
                "run_command",
                {"command": command, "timeout_seconds": timeout_seconds},
            )
            _validate_command_input(command)
            _validate_command(command)
            timeout = _command_timeout(timeout_seconds)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error
        await self._ensure_open()
        process = await self._start_process(command)
        cleanup_succeeded = True
        try:
            await asyncio.wait_for(asyncio.shield(process.wait_task), timeout)
        except asyncio.TimeoutError as error:
            try:
                await _stop_process(process, force=True)
            except BaseException:
                cleanup_succeeded = False
                raise
            del error
            process.final_status = "timeout"
            return _command_result(process)
        except asyncio.CancelledError as cancellation:
            try:
                await _finish_task(asyncio.create_task(
                    _stop_process(process, force=True)
                ))
            except BaseException as cleanup_error:
                cleanup_succeeded = False
                _logger.exception(
                    "cancelled local command cleanup failed: command=%s",
                    process.command_id,
                    exc_info=cleanup_error,
                )
                raise cancellation from cleanup_error
            raise
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                cleanup_succeeded = False
            raise
        finally:
            if cleanup_succeeded:
                await self._forget_process(process.command_id)
        return _command_result(process)

    async def start_command(self, command: str) -> str:
        self._reject_read_only()
        try:
            validate_request_size("start_command", {"command": command})
            _validate_command_input(command)
            _validate_command(command)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error
        await self._ensure_open()
        async with self._process_lock:
            if len(self._processes) + self._starting_background >= 256:
                raise WorkspaceOperationRejected(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
            self._starting_background += 1
        try:
            process = await self._start_process(command)
        finally:
            async with self._process_lock:
                self._starting_background -= 1
        return _bound_output(f"command_id: {process.command_id}\nstatus: running")

    async def check_command(self, command_id: str) -> str:
        self._reject_read_only()
        validate_request_size("check_command", {"command_id": command_id})
        process = await self._get_process(command_id)
        if process is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if process.wait_task.done():
            process.wait_task.result()
            status = "exited"
        else:
            status = "running"
        return _command_result(process, status=status)

    async def stop_command(self, command_id: str) -> str:
        self._reject_read_only()
        try:
            validate_request_size("stop_command", {"command_id": command_id})
            if not command_id:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error
        process = await self._get_process(command_id)
        if process is None:
            raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID)
        await _stop_process(process, force=False)
        return _command_result(process)

    async def close(self) -> None:
        async with self._state_lock:
            if self._state == "CLOSED":
                return
            if self._close_task is None:
                self._state = "CLOSING"
                self._close_task = asyncio.create_task(
                    self._close_impl(),
                    name="local-sandbox-close",
                )
            task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                await _finish_task(task)
            except BaseException as cleanup_error:
                if not isinstance(cleanup_error, asyncio.CancelledError):
                    _logger.exception(
                        "cancelled local sandbox cleanup failed",
                        exc_info=cleanup_error,
                    )
                raise cancellation from cleanup_error
            raise cancellation

    async def _close_impl(self) -> None:
        cleanup_error: BaseException | None = None
        try:
            await self._wait_operations()
        except BaseException as error:
            cleanup_error = error
            _logger.exception("local sandbox cleanup wait failed")
        async with self._process_lock:
            processes = tuple(self._processes.values())
            stdio_processes = tuple(self._stdio_processes)
            pending_processes = tuple(self._pending_processes.items())
        for process in stdio_processes:
            try:
                await process.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
                _logger.exception("local stdio process cleanup failed")
        for process in processes:
            try:
                await _stop_process(process, force=True)
            except BaseException as error:
                cleanup_error = cleanup_error or error
                _logger.exception(
                    "local sandbox process cleanup failed: command=%s",
                    process.command_id,
                )
            else:
                async with self._process_lock:
                    if self._processes.get(process.command_id) is process:
                        self._processes.pop(process.command_id, None)
        for process_key, (process, job) in pending_processes:
            try:
                await _terminate_unregistered_process(process, job)
            except BaseException as error:
                cleanup_error = cleanup_error or error
                _logger.exception("local pending process cleanup failed")
                if process.returncode is not None:
                    self._pending_processes.pop(process_key, None)
            else:
                self._pending_processes.pop(process_key, None)
        try:
            if cleanup_error is not None:
                raise cleanup_error
            async with self._state_lock:
                self._state = "CLOSED"
            _logger.debug("local sandbox session closed")
        except BaseException as error:
            async with self._state_lock:
                self._state = "LOST"
            _logger.exception("local sandbox session cleanup failed")
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error

    async def _wait_operations(self) -> None:
        operations = (*self._operations, *self._starting_operations)
        await asyncio.gather(
            *(asyncio.shield(task) for task in operations),
            return_exceptions=True,
        )

    async def _ensure_open(self) -> None:
        await self._initialize()
        async with self._state_lock:
            if self._state != "OPEN":
                code = (
                    ErrorCode.SANDBOX_SESSION_LOST
                    if self._state == "LOST"
                    else ErrorCode.SANDBOX_SESSION_CLOSED
                )
                raise AIError(code)

    def _reject_read_only(self) -> None:
        if self._read_policy is not None:
            raise WorkspaceOperationRejected(ErrorCode.AUTHORIZATION_DENIED)

    def _authorize_read(
        self,
        path: str,
        *,
        resource_key: str | None = None,
        directory: bool | None = False,
    ) -> None:
        if self._read_policy is None and resource_key is None:
            return
        if directory is None:
            allowed = self._visible_read_path(
                path,
                resource_key=resource_key,
                directory=False,
            ) or self._visible_read_path(
                path,
                resource_key=resource_key,
                directory=True,
            )
        else:
            allowed = self._visible_read_path(
                path,
                resource_key=resource_key,
                directory=directory,
            )
        if not allowed:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    def _visible_read_path(
        self,
        path: str,
        *,
        resource_key: str | None = None,
        directory: bool,
    ) -> bool:
        try:
            resource = self._resources[resource_key][1] if resource_key is not None else None
            if resource is not None and resource.files is not None:
                if directory:
                    prefix = "" if path == "." else path + "/"
                    return any(
                        member.startswith(prefix)
                        and (self._read_policy is None or self._read_policy.allows(
                            member, resource_key=resource_key,
                        ))
                        for member in resource.files
                    )
                if path not in resource.files:
                    return False
            if self._read_policy is None:
                return True
            return (
                self._read_policy.may_descend(path, resource_key=resource_key)
                if directory
                else self._read_policy.allows(path, resource_key=resource_key)
            )
        except ValueError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

    def _read_target(
        self,
        path: str,
        *,
        allow_missing: bool = False,
        directory: bool | None = False,
    ) -> tuple[Path, str, str | None]:
        resource_key: str | None = None
        base = self._root
        if (
            isinstance(path, str)
            and Path(path).is_absolute()
        ):
            candidate = Path(path)
            resource_matches = tuple(
                (key, source)
                for key, (source, _resource) in self._resources.items()
                if source is not None and _inside_or_equal(source, candidate)
            )
            if resource_matches:
                resource_key, base = max(
                    resource_matches,
                    key=lambda item: len(item[1].parts),
                )
            elif _inside_or_equal(self._root, candidate):
                base = self._root
            else:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            try:
                relative = candidate.relative_to(base).as_posix()
            except ValueError as error:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error
            normalized = _normalize_path(relative)
        else:
            normalized = _normalize_path(path)
        self._authorize_read(
            normalized,
            resource_key=resource_key,
            directory=directory,
        )
        target = self._resolve_path(
            normalized,
            allow_missing=allow_missing,
            base=base,
        )
        if (
            self._read_policy is None
            and directory
            and base == self._root
            and (base / normalized).is_symlink()
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return target, normalized, resource_key

    def _ensure_open_sync(self) -> None:
        if self._state != "OPEN":
            code = (
                ErrorCode.SANDBOX_SESSION_LOST
                if self._state == "LOST"
                else ErrorCode.SANDBOX_SESSION_CLOSED
            )
            raise AIError(code)

    async def _run_sync(self, operation: Callable[[], _SyncResult]) -> _SyncResult:
        await self._ensure_open()
        async with self._state_lock:
            if self._state != "OPEN":
                code = (
                    ErrorCode.SANDBOX_SESSION_LOST
                    if self._state == "LOST"
                    else ErrorCode.SANDBOX_SESSION_CLOSED
                )
                raise AIError(code)
            task = asyncio.create_task(asyncio.to_thread(operation))
            self._operations.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await _finish_task(task)
            except BaseException as error:
                _logger.exception(
                    "cancelled local sandbox operation completed with an error",
                    exc_info=error,
                )
            raise
        finally:
            self._operations.discard(task)

    def _resolve_path(
        self,
        path: str,
        *,
        allow_missing: bool,
        base: Path | None = None,
    ) -> Path:
        self._ensure_open_sync()
        trusted_root = self._root if base is None else base
        candidate = trusted_root / path
        if candidate != trusted_root:
            _check_parent_chain(trusted_root, candidate.parent)
        if self._read_policy is not None and candidate.is_symlink():
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        try:
            if candidate.is_symlink():
                link = os.readlink(candidate)
                if _is_absolute_link(link):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            resolved = candidate.resolve(strict=False)
        except AIError:
            raise
        except RuntimeError as error:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
        if not _inside(trusted_root, resolved):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if not allow_missing and not resolved.exists():
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return resolved if candidate.is_symlink() else candidate

    def _file_path(self, path: str, *, write: bool, allow_missing_parent: bool = False) -> Path:
        if write and _is_protected(self._workspace, path):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        candidate = self._resolve_path(path, allow_missing=write)
        resolved = candidate.resolve(strict=False)
        if write and _is_protected(self._workspace, _relative(self._root, resolved)):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if write:
            parent = candidate.parent
            if allow_missing_parent:
                while parent != self._root and not parent.exists():
                    parent = parent.parent
            if not parent.is_dir():
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if candidate.exists():
            info = candidate.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif not write:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return candidate


    async def _start_process(self, command: str, *, environment: Mapping[str, str] | None = None) -> "_ProcessState":
        started = asyncio.get_running_loop().create_future()
        async with self._process_lock:
            if self._state != "OPEN":
                raise AIError(_session_state_error(self._state))
            self._starting_operations.add(started)
        try:
            return await self._spawn_process(command, environment=environment)
        finally:
            started.set_result(None)
            self._starting_operations.discard(started)

    async def _spawn_process(self, command: str, *, environment: Mapping[str, str] | None = None) -> "_ProcessState":
        if os.name != "posix":
            raise WorkspaceOperationRejected(ErrorCode.SANDBOX_UNAVAILABLE,
                                              safe_details={"reason": "posix_execution_required"})
        await self._ensure_open()
        environment = {**self._environment, **(environment or {})}
        job: _WindowsJob | None = None
        executable = "/bin/sh"
        process_kwargs: dict[str, object] = {
            "cwd": str(self._root),
            "env": environment,
            "stdin": DEVNULL,
            "stdout": PIPE,
            "stderr": PIPE,
        }
        if os.name == "nt":
            process_kwargs["creationflags"] = 0x00000004
        else:
            process_kwargs["start_new_session"] = True

        async def create_process() -> asyncio.subprocess.Process:
            return await asyncio.create_subprocess_exec(
                executable,
                "-c",
                command,
                **process_kwargs,
            )

        process_task = asyncio.create_task(
            create_process(),
            name="local-sandbox-process-start",
        )
        try:
            process = await asyncio.shield(process_task)
        except asyncio.CancelledError as cancellation:
            try:
                process = await _finish_task(process_task)
            except BaseException:
                if job is not None:
                    job.close()
                raise
            process_key = id(process)
            self._pending_processes[process_key] = (process, job)
            try:
                await _finish_task(asyncio.create_task(
                    _terminate_unregistered_process(process, job)
                ))
            except BaseException as cleanup_error:
                _logger.exception(
                    "cancelled local process cleanup failed",
                    exc_info=cleanup_error,
                )
                if process.returncode is not None:
                    self._pending_processes.pop(process_key, None)
                raise cancellation from cleanup_error
            self._pending_processes.pop(process_key, None)
            raise cancellation
        except (OSError, ValueError) as error:
            if job is not None:
                job.close()
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        process_key = id(process)
        self._pending_processes[process_key] = (process, job)
        if job is not None:
            try:
                job.assign_and_resume(process.pid)
            except BaseException as error:
                try:
                    await _terminate_unregistered_process(process, job)
                except BaseException as cleanup_error:
                    if process.returncode is not None:
                        self._pending_processes.pop(process_key, None)
                    if isinstance(error, asyncio.CancelledError):
                        raise error from cleanup_error
                    raise cleanup_error from error
                self._pending_processes.pop(process_key, None)
                if isinstance(error, asyncio.CancelledError):
                    raise error
                if isinstance(error, AIError):
                    raise error
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        command_id = uuid.uuid4().hex
        state = _ProcessState(command_id, command, process, job=job)
        registered = False
        cleanup_attempted = False
        try:
            async with self._process_lock:
                if self._state == "OPEN":
                    self._processes[command_id] = state
                    registered = True
            if registered:
                self._pending_processes.pop(process_key, None)
                return state
            self._processes[command_id] = state
            self._pending_processes.pop(process_key, None)
            cleanup_attempted = True
            try:
                await _stop_process_state(state, force=True)
            except BaseException as cleanup_error:
                _logger.exception(
                    "local process cleanup after session close failed: command=%s",
                    command_id,
                    exc_info=cleanup_error,
                )
            else:
                self._processes.pop(command_id, None)
            raise AIError(_session_state_error(self._state))
        except BaseException:
            if not registered and not cleanup_attempted:
                self._processes[command_id] = state
                self._pending_processes.pop(process_key, None)
                cleanup_attempted = True
                try:
                    await _stop_process_state(state, force=True)
                except BaseException as cleanup_error:
                    _logger.exception(
                        "local unregistered process cleanup failed: command=%s",
                        command_id,
                        exc_info=cleanup_error,
                    )
                else:
                    self._processes.pop(command_id, None)
            raise

    async def _get_process(self, command_id: str) -> "_ProcessState | None":
        if not isinstance(command_id, str) or not command_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        await self._ensure_open()
        async with self._process_lock:
            return self._processes.get(command_id)

    async def _forget_process(self, command_id: str) -> None:
        async with self._process_lock:
            self._processes.pop(command_id, None)


class _LocalStdioProcess:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        job: "_WindowsJob | None",
        on_close: Callable[["_LocalStdioProcess"], None],
    ) -> None:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        self._process = process
        self._job = job
        self._on_close = on_close
        self._stdin_closed = False
        self._state = "OPEN"
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process.stderr),
            name="local-sandbox-stdio-stderr",
        )

    async def write_stdin(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._state != "OPEN" or self._stdin_closed:
            raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
        stdin = self._process.stdin
        if stdin is None:
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST)
        try:
            stdin.write(data)
            await stdin.drain()
        except (BrokenPipeError, ConnectionError, OSError) as error:
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST) from error

    async def read_stdout(self, max_bytes: int = 65536) -> bytes:
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self._state != "OPEN":
            raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
        stdout = self._process.stdout
        if stdout is None:
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST)
        try:
            return await stdout.read(max_bytes)
        except OSError as error:
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST) from error

    async def close_stdin(self) -> None:
        async with self._lock:
            if self._stdin_closed:
                return
            self._stdin_closed = True
            stdin = self._process.stdin
            if stdin is not None and not stdin.is_closing():
                stdin.close()

    async def close(self) -> None:
        async with self._lock:
            if self._state == "CLOSED":
                return
            if self._close_task is None:
                self._state = "CLOSING"
                self._close_task = asyncio.create_task(
                    self._close_impl(),
                    name="local-sandbox-stdio-close",
                )
            task = self._close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                await _finish_task(task)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise cancellation
        except BaseException:
            async with self._lock:
                if self._close_task is task:
                    self._close_task = None
                self._state = "OPEN"
            raise

    async def _close_impl(self) -> None:
        await self.close_stdin()
        self._stderr_task.cancel()
        await asyncio.gather(self._stderr_task, return_exceptions=True)
        try:
            await _terminate_unregistered_process(self._process, self._job)
            stdin = self._process.stdin
            if stdin is not None:
                # Discard buffered input even if a detached reader retained the pipe.
                if stdin.transport.get_write_buffer_size():
                    stdin.transport.abort()
                try:
                    await stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        except BaseException as error:
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error
        self._job = None
        self._state = "CLOSED"
        self._on_close(self)

    @staticmethod
    async def _drain_stderr(stderr: asyncio.StreamReader) -> None:
        retained = 0
        try:
            while True:
                chunk = await stderr.read(4096)
                if not chunk:
                    return
                remaining = 64 * 1024 - retained
                if remaining > 0:
                    visible = chunk[:remaining]
                    if environ.debug:
                        _logger.debug(
                            "local stdio stderr: %s",
                            visible.decode("utf-8", "replace"),
                        )
                    retained += len(visible)
        except asyncio.CancelledError:
            raise
        except OSError:
            return


def _local_stdio_command_args(
    args: Sequence[str | WorkspaceResourcePath],
    resources: Sequence[WorkspaceResource],
) -> tuple[str, ...]:
    by_id = {resource.id: resource for resource in resources}
    result: list[str] = []
    for argument in args:
        if isinstance(argument, str):
            if "\x00" in argument:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            result.append(argument)
            continue
        if not isinstance(argument, WorkspaceResourcePath):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        resource = by_id.get(argument.resource_id)
        if resource is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source: Path | None = None
        if resource.files is not None:
            source = resource.files.get(argument.path)
        elif resource.source is not None:
            candidate = resource.source.joinpath(*PurePosixPath(argument.path).parts)
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(resource.source.resolve())
            except (OSError, RuntimeError, ValueError) as error:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            if resolved.is_file():
                source = resolved
        if source is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        result.append(str(source))
    return tuple(result)


def _normalize_root(root: Path) -> Path:
    if not isinstance(root, Path):
        raise TypeError("sandbox root must be a Path")
    if not root.is_absolute():
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        value = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    if not value.is_dir():
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return value


def _session_state_error(state: str) -> ErrorCode:
    return (
        ErrorCode.SANDBOX_SESSION_LOST
        if state == "LOST"
        else ErrorCode.SANDBOX_SESSION_CLOSED
    )


def _validate_resources(
    workspace: Workspace,
    resources: tuple[WorkspaceResource, ...],
) -> tuple[WorkspaceResource, ...]:
    if not isinstance(resources, tuple):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    seen: set[str] = set()
    values: list[WorkspaceResource] = []
    root = workspace.root
    storage_root = workspace.storage_root
    for resource in resources:
        if not isinstance(resource, WorkspaceResource) or resource.id in seen:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = resource.source
        if source is None:
            seen.add(resource.id)
            values.append(resource)
            continue
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        if (
            not resolved.is_dir()
            or source.is_symlink()
            or resolved == root
            or _inside(resolved, root)
            or resolved == storage_root
            or _inside(resolved, storage_root)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        seen.add(resource.id)
        values.append(resource)
    return tuple(values)


def _normalize_path(path: str) -> str:
    value = normalize_workspace_input_path(path)
    if os.name == "nt":
        _validate_windows_path(value)
    return value


def _inside(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


def _inside_or_equal(root: Path, target: Path) -> bool:
    return _inside(root, target)


def _child_display(display: str, child: Path, parent: Path) -> str:
    if child == parent:
        return display
    relative = child.relative_to(parent).as_posix()
    return relative if display == "." else f"{display}/{relative}"


def _relative(root: Path, target: Path) -> str:
    return target.relative_to(root).as_posix()


def _is_absolute_link(value: str) -> bool:
    if os.name == "nt":
        import ntpath

        drive, _ = ntpath.splitdrive(value)
        return bool(drive) or ntpath.isabs(value)
    return Path(value).is_absolute()


def _check_parent_chain(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as error:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED) from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def _is_protected(workspace: Workspace, path: str) -> bool:
    normalized = path.replace("\\", "/")
    if (
        workspace.is_storage_path(normalized)
        or normalized == ".git"
        or normalized.startswith(".git/")
    ):
        return True
    for part in normalized.split("/"):
        if part.startswith("secrets"):
            return True
        if any(fnmatch.fnmatchcase(part, pattern) for pattern in _PROTECTED_NAMES):
            return True
    return False


@dataclass
class _WalkState:
    failed: bool = False


def _search_bounded_lines(
    path: Path, logical: str, expression: re.Pattern[str], max_bytes: int,
    deadline: float, limit: int,
) -> tuple[list[str], bool, int]:
    rows: list[str] = []
    scanned = 0
    line_number = 0
    retained = ""
    dropping = False
    incomplete = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending_cr = ""
    with path.open("rb") as stream:
        state = _state(logical, os.fstat(stream.fileno()))
        while True:
            if time.monotonic() > deadline or scanned >= max_bytes:
                incomplete = True
                break
            chunk = stream.read(min(64 * 1024, max_bytes - scanned))
            if scanned == 0 and not _facts(chunk[:8192], logical, os.fstat(stream.fileno()).st_size)["is_text"]:
                _check_state(path, logical, stream, state)
                return [], False, len(chunk)
            scanned += len(chunk)
            value = pending_cr + decoder.decode(chunk, final=not chunk)
            pending_cr = ""
            if chunk and value.endswith("\r"):
                value, pending_cr = value[:-1], "\r"
            # Retain at most one output-sized line. Longer lines are explicitly uncovered.
            for fragment in value.splitlines(keepends=True):
                ended = fragment.endswith(("\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"))
                if not dropping:
                    if len(retained) + len(fragment) > _MAX_CONTENT_CHARS:
                        dropping = incomplete = True
                        retained = ""
                    else:
                        retained += fragment
                if ended:
                    line_number += 1
                    if not dropping and expression.search(retained.rstrip("\r\n")):
                        rows.append(f"{logical}:{line_number}:{retained.rstrip(chr(13) + chr(10))}")
                    retained = ""
                    dropping = False
                    if len(rows) >= limit:
                        incomplete = True
                        break
            if len(rows) >= limit:
                break
            if not chunk:
                if retained and not dropping and expression.search(retained):
                    rows.append(f"{logical}:{line_number + 1}:{retained}")
                break
        _check_state(path, logical, stream, state)
    return rows, incomplete, scanned


def _stream_edit(
    root: Path, path: Path, logical: str, old_text: str, new_text: str,
    expected_hash: str | None,
) -> str:
    needle, replacement = old_text.encode("utf-8"), new_text.encode("utf-8")
    original = hashlib.sha256()
    updated = hashlib.sha256()
    source_state: str | None = None
    deadline = time.monotonic() + _MAX_FILE_SECONDS

    def prepare(output: BinaryIO) -> None:
        nonlocal source_state
        try:
            with path.open("rb") as source:
                info = os.fstat(source.fileno())
                if not stat.S_ISREG(info.st_mode):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if info.st_size > _MAX_EDIT_BYTES:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "edit_scan_byte_limit"})
                source_state = _state(logical, info)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
                pending = b""
                count = scanned = written = 0

                def emit(value: bytes) -> None:
                    nonlocal written
                    written += len(value)
                    if written > _MAX_EDIT_BYTES:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "edit_staging_byte_limit"})
                    output.write(value)
                    updated.update(value)

                while True:
                    if time.monotonic() > deadline:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "edit_scan_time_limit"})
                    chunk = source.read(64 * 1024)
                    scanned += len(chunk)
                    if scanned > _MAX_EDIT_BYTES:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "edit_scan_byte_limit"})
                    original.update(chunk)
                    decoded = decoder.decode(chunk, final=not chunk)
                    if "\x00" in decoded:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_text"})
                    pending += chunk
                    while True:
                        index = pending.find(needle)
                        if index < 0:
                            break
                        count += 1
                        emit(pending[:index])
                        emit(replacement if count == 1 else needle)
                        pending = pending[index + len(needle):]
                    if not chunk:
                        emit(pending)
                        break
                    keep = len(needle) - 1
                    if len(pending) > keep:
                        end = len(pending) - keep
                        emit(pending[:end])
                        pending = pending[end:]
                if expected_hash is not None:
                    _check_expected_digest(original.hexdigest(), expected_hash)
                if count != 1:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "no_match" if count == 0 else "multiple_matches"})
                _check_state(path, logical, source, source_state)
        except FileNotFoundError as error:
            code = ErrorCode.STORAGE_CONFLICT if expected_hash is not None else ErrorCode.STORAGE_NOT_FOUND
            raise WorkspaceOperationRejected(code) from error
        except UnicodeDecodeError as error:
            raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "invalid_utf8"}) from error
        except AIError as error:
            raise WorkspaceOperationRejected.from_error(error) from error
        except OSError as error:
            raise WorkspaceOperationRejected(ErrorCode.STORAGE_UNAVAILABLE) from error

    def check_before_publish() -> None:
        try:
            if source_state is None or _state(logical, path.stat()) != source_state:
                raise WorkspaceOperationRejected(ErrorCode.STORAGE_CONFLICT, safe_details={"reason": "file_changed"})
        except FileNotFoundError as error:
            raise WorkspaceOperationRejected(ErrorCode.STORAGE_CONFLICT) from error

    _atomic_stream_write(root, path, prepare, before_publish=check_before_publish)
    return updated.hexdigest()


def _read_optional_hash(path: Path, logical: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            while chunk := stream.read(64 * 1024):
                digest.update(chunk)
            if _file_identity(stream.fileno()) != identity or _path_identity(path) != identity:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return digest.hexdigest()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _file_identity(fd: int) -> tuple[int, int, int, int]:
    info = os.fstat(fd)
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
    )


def _path_identity(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
    )


def _validate_text(value: str) -> None:
    if not isinstance(value, str) or len(value) > _MAX_WRITE_CHARS or "\x00" in value:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _validate_windows_path(path: str) -> None:
    if path == ".":
        return
    for part in path.split("/"):
        if (
            not part
            or part[-1] in {".", " "}
            or ":" in part
            or any(character in '<>"|?*' for character in part)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _validate_expected_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def _validate_glob(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_SEARCH_PATTERN_CHARS
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
        or any(part == ".." for part in value.split("/"))
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _check_expected_digest(actual: str | None, expected: str) -> None:
    if actual is None or actual != expected:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _create_temp_file(parent: Path) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(8):
        target = parent / f"{Workspace.STORAGE_DIR_NAME}-{uuid.uuid4().hex}"
        try:
            return os.open(str(target), flags, 0o666), target
        except FileExistsError:
            continue
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    raise AIError(ErrorCode.STORAGE_UNAVAILABLE)


def _copy_windows_security(source: Path, target: Path) -> None:
    if os.name != "nt":
        return
    from ctypes import wintypes

    security_info = 0x00000001 | 0x00000002 | 0x00000004
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    get_file_security = advapi32.GetFileSecurityW
    get_file_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_file_security.restype = wintypes.BOOL
    required = wintypes.DWORD()
    get_file_security(
        str(source),
        security_info,
        None,
        0,
        ctypes.byref(required),
    )
    if not required.value:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
    descriptor = ctypes.create_string_buffer(required.value)
    if not get_file_security(
        str(source),
        security_info,
        descriptor,
        required,
        ctypes.byref(required),
    ):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
    set_file_security = advapi32.SetFileSecurityW
    set_file_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    set_file_security.restype = wintypes.BOOL
    if not set_file_security(str(target), security_info, descriptor):
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)


def _atomic_write(root: Path, path: Path, content: str) -> None:
    def write(stream: BinaryIO) -> None:
        stream.write(content.encode("utf-8"))
    created: list[tuple[Path, int, int]] = []
    try:
        current = root
        for part in path.parent.relative_to(root).parts:
            current = current / part
            _check_parent_chain(root, current)
            try:
                current.mkdir()
            except FileExistsError:
                if current.is_symlink() or not current.is_dir():
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            else:
                info = current.lstat()
                created.append((current, info.st_dev, info.st_ino))
        _atomic_stream_write(root, path, write)
    except BaseException as error:
        for directory, device, inode in reversed(created):
            try:
                _check_parent_chain(root, directory)
                info = directory.lstat()
                if (info.st_dev, info.st_ino) == (device, inode) and stat.S_ISDIR(info.st_mode):
                    directory.rmdir()
            except (OSError, AIError):
                pass
        if isinstance(error, OSError):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        raise


def _atomic_stream_write(
    root: Path, path: Path, write: Callable[[BinaryIO], None], *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    _check_parent_chain(root, path.parent)
    parent = path.parent
    if not parent.is_dir():
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
    mode = None
    owner: tuple[int, int] | None = None
    attributes: dict[str, bytes] = {}
    if path.exists():
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if info.st_nlink > 1:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        mode = stat.S_IMODE(info.st_mode)
        owner = (info.st_uid, info.st_gid)
        if hasattr(os, "listxattr"):
            try:
                attributes = {
                    name: os.getxattr(path, name)
                    for name in os.listxattr(path)
                }
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    fd, temporary_path = _create_temp_file(parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
            if owner is not None:
                if os.name == "nt":
                    _copy_windows_security(path, temporary_path)
                else:
                    try:
                        os.fchown(stream.fileno(), *owner)
                    except (AttributeError, OSError) as error:
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            if mode is not None and os.name != "nt":
                os.fchmod(stream.fileno(), mode)
        if attributes and hasattr(os, "setxattr"):
            try:
                for name, value in attributes.items():
                    os.setxattr(temporary_path, name, value)
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if before_publish is not None:
            before_publish()
        os.replace(temporary_path, path)
    except AIError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _prepare_lock_root(lock_dir: Path) -> None:
    current = lock_dir
    while True:
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        if current.exists() or current.parent == current:
            break
        current = current.parent
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error


def _file_lock(lock_dir: Path, logical: str) -> FileLock:
    _prepare_lock_root(lock_dir)
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    return FileLock(str(lock_dir / f"{digest}.lock"))


def _bound_output(value: str) -> str:
    if len(value) <= _MAX_RENDERED_CHARS:
        return value
    return _bound_output_with_marker(value, "\n[truncated]")


def _bound_output_with_marker(value: str, marker: str) -> str:
    if not value.endswith(marker):
        value += marker if value else marker.lstrip("\n")
    if len(value) <= _MAX_RENDERED_CHARS:
        return value
    return value[: _MAX_RENDERED_CHARS - len(marker)] + marker


def _write_result(path: str, value: str, *, status: str = "written") -> str:
    return _bound_output(f"path: {path}\nstatus: {status}\nhash: {value}")


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    if (
        not isinstance(pattern, str)
        or not pattern
        or len(pattern) > _MAX_SEARCH_PATTERN_CHARS
        or "\x00" in pattern
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        return re.compile(pattern)
    except re.error as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _validate_command(command: str) -> None:
    _validate_command_input(command)
    stripped = command.strip()
    if (
        _INTERACTIVE_COMMANDS.match(stripped)
        or _SUDO_COMMAND.match(stripped)
        or _REMOTE_COMMANDS.match(stripped)
    ):
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = []
    if tokens and tokens[0] in _DANGEROUS_COMMANDS:
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def _validate_command_input(command: str) -> None:
    if not isinstance(command, str) or "\x00" in command:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        command.encode(sys.getfilesystemencoding(), errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


def _command_timeout(value: float | None) -> float:
    timeout = _DEFAULT_COMMAND_TIMEOUT if value is None else value
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
        or timeout != timeout
        or timeout in {float("inf"), float("-inf")}
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return float(timeout)


def _command_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not _secret_environment_key(key)
    }


def _secret_environment_key(key: str) -> bool:
    return key.startswith(
        (
            "ANTHROPIC_",
            "GATEWAY_",
            "GEMINI_",
            "GOOGLE_",
            "OPENAI_",
            "OPENROUTER_",
        )
    ) or key == "PYDANTIC_AI_GATEWAY_API_KEY"


__all__ = ["LocalWorkspaceBackend"]
