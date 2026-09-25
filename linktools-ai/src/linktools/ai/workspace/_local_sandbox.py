#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local execution backend.

The local backend provides the sandbox contract and lifecycle guarantees; it
is deliberately not a security sandbox.  Agent code still only receives the
logical file and command operations exposed by ``SandboxSession``.
"""

import asyncio
import codecs
import ctypes
import fnmatch
import heapq
import hashlib
import io
import os
import re
import shlex
import stat
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from subprocess import DEVNULL, PIPE

from filelock import FileLock
from linktools.core import environ

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._sandbox import (
    SandboxOperationRejected,
    ReadOnlySandboxPolicy,
    SandboxResource,
    SandboxResourcePath,
    SandboxSession,
    SandboxStdioProcess,
    normalize_workspace_input_path,
)
from ._root import Workspace
from ._sandbox_protocol import validate_request_size
from ._local_process import (
    _ProcessState,
    _WindowsJob,
    _command_result,
    _stop_process,
    _stop_process_state,
    _terminate_unregistered_process,
)

_logger = environ.get_logger("ai.workspace.local_sandbox")
_MAX_RENDERED_CHARS = 50_000
_MAX_CONTENT_CHARS = 50_000
_MAX_READ_LINES = 2_000
_MAX_ITEMS = 1_000
_MAX_SEARCH_PATTERN_CHARS = 1_000
_MAX_WRITE_CHARS = 10 * 1024 * 1024
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


class LocalSandbox:
    """Open local sessions rooted at the caller-provided directory."""

    def __init__(self, *, read_policy: ReadOnlySandboxPolicy | None = None) -> None:
        if read_policy is not None and not isinstance(
            read_policy,
            ReadOnlySandboxPolicy,
        ):
            raise TypeError("read_policy must be ReadOnlySandboxPolicy")
        self._read_policy = read_policy

    def stdio_execution_policy(self) -> Mapping[str, JsonValue]:
        """Describe LocalSandbox stdio as an explicit host process boundary."""
        if self._read_policy is not None:
            raise AIError(
                ErrorCode.SANDBOX_UNAVAILABLE,
                safe_details={"reason": "local_stdio_read_policy_unsupported"},
            )
        return {"version": 1, "boundary": "host-stdio"}

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        policy = self._read_policy
        normalized_root = _normalize_root(root)
        workspace = Workspace(normalized_root, {})
        normalized_resources = _validate_resources(workspace, resources)
        _logger.debug(
            "opening local sandbox session: root=%s resources=%s",
            normalized_root,
            tuple(resource.id for resource in normalized_resources),
        )
        lock_root = workspace.locks_root
        return _LocalSandboxSession(
            normalized_root,
            normalized_resources,
            lock_root=lock_root,
            read_policy=policy,
            workspace=workspace,
        )




class _LocalSandboxSession:
    def __init__(
        self,
        root: Path,
        resources: tuple[SandboxResource, ...],
        *,
        lock_root: Path | None = None,
        read_policy: ReadOnlySandboxPolicy | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        self._root = root
        self._workspace = workspace if workspace is not None else Workspace(root, {})
        self._resources = {
            resource.id: resource.source.resolve()
            for resource in resources
            if resource.source is not None
        }
        self._resource_ids = frozenset(resource.id for resource in resources)
        self._lock_root = lock_root or self._workspace.locks_root
        self._read_policy = read_policy
        self._environment = _command_environment()
        self._state = "OPEN"
        self._state_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._processes: dict[str, _ProcessState] = {}
        self._stdio_processes: set[_LocalStdioProcess] = set()
        self._pending_processes: dict[
            int, tuple[asyncio.subprocess.Process, _WindowsJob | None]
        ] = {}
        self._operations: set[asyncio.Task[object]] = set()
        self._starting_tasks: set[asyncio.Task[object]] = set()
        self._starting_background = 0
        self._process_lock = asyncio.Lock()

    def resource_path(self, resource_id: str) -> "str | None":
        self._ensure_open_sync()
        if not isinstance(resource_id, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if resource_id not in self._resource_ids:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = self._resources.get(resource_id)
        if source is None:
            return None
        if (
            self._read_policy is not None
            and not self._read_policy.may_descend(
                ".",
                resource_key=resource_id,
            )
        ):
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
        args: "Sequence[str | SandboxResourcePath]" = (),
        *,
        resources: "Sequence[SandboxResource]" = (),
    ) -> SandboxStdioProcess:
        if self._read_policy is not None:
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

        selected_resources = _validate_resources(
            self._workspace,
            tuple(resources),
        )
        command_args = _local_stdio_command_args(args, selected_resources)
        current = asyncio.current_task()
        async with self._process_lock:
            if self._state != "OPEN":
                raise AIError(_session_state_error(self._state))
            if current is not None:
                self._starting_tasks.add(current)

        job: _WindowsJob | None = None
        process: asyncio.subprocess.Process | None = None
        process_key: int | None = None
        try:
            if os.name == "nt":
                job = _WindowsJob.create()
            process_kwargs: dict[str, object] = {
                "cwd": str(self._root),
                "env": dict(self._environment),
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
                    process = await asyncio.shield(process_task)
                except BaseException:
                    if job is not None:
                        job.close()
                    raise
                process_key = id(process)
                self._pending_processes[process_key] = (process, job)
                try:
                    await _terminate_unregistered_process(process, job)
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
            if current is not None:
                self._starting_tasks.discard(current)

    async def canonicalize_path(self, path: str) -> str:
        """Normalize one logical Workspace path without applying operation policy."""
        def operation() -> str:
            return _normalize_path(path)

        return await self._run_sync(operation)

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        if max_bytes is not None and (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        def operation() -> bytes:
            target, _display, _resource_key = self._read_target(path)
            try:
                info = target.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if max_bytes is None:
                    return target.read_bytes()
                with target.open("rb") as stream:
                    value = stream.read(max_bytes + 1)
                if len(value) > max_bytes:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                return value
            except AIError:
                raise
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

        return await self._run_sync(operation)

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        selected_limit = _MAX_READ_LINES if limit is None else limit
        if (
            not isinstance(selected_limit, int)
            or isinstance(selected_limit, bool)
            or selected_limit <= 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        selected_limit = min(selected_limit, _MAX_READ_LINES)
        validate_request_size(
            "read_file",
            {"path": path, "offset": offset, "limit": limit},
        )
        def operation() -> str:
            target, display, _resource_key = self._read_target(path)
            try:
                info = target.stat()
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            if not stat.S_ISREG(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            binary_size = _binary_file_size(target)
            if binary_size is not None:
                return _bound_output(
                    f"[Binary file: {binary_size} bytes. "
                    "Use a binary-aware tool to inspect.]"
                )
            digest, line_count, visible, selected_truncated = _read_selected_lines(
                target,
                display,
                offset,
                selected_limit,
            )
            if offset >= line_count and line_count:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            rendered = _render_read(
                display,
                digest,
                line_count,
                offset,
                visible,
                selected_truncated,
            )
            return _bound_output(rendered)

        return await self._run_sync(operation)

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
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
            raise SandboxOperationRejected.from_error(error) from error

        def operation() -> str:
            try:
                target = self._file_path(normalized, write=True)
            except AIError as error:
                raise SandboxOperationRejected.from_error(error) from error
            with _file_lock(self._lock_root, _relative(self._root, target)):
                if expected is not None:
                    try:
                        current_hash = _read_optional_hash(target, normalized)
                        _check_expected_digest(current_hash, expected)
                    except AIError as error:
                        raise SandboxOperationRejected.from_error(error) from error
                _atomic_write(self._root, target, content)
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            return _write_result(normalized, digest)

        return await self._run_sync(operation)

    async def edit_file(
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
                "edit_file",
                {
                    "path": path,
                    "old_text": old_text,
                    "new_text": new_text,
                    "expected_hash": expected_hash,
                },
            )
            _validate_text(old_text)
            _validate_text(new_text)
        except AIError as error:
            raise SandboxOperationRejected.from_error(error) from error

        def operation() -> str:
            try:
                target = self._file_path(normalized, write=True)
            except AIError as error:
                raise SandboxOperationRejected.from_error(error) from error
            with _file_lock(self._lock_root, _relative(self._root, target)):
                try:
                    current = _read_optional_text(target, normalized)
                    _check_expected_hash(current, expected)
                    if current is None:
                        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                    if current.count(old_text) != 1:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    updated = current.replace(old_text, new_text)
                    _validate_text(updated)
                except AIError as error:
                    raise SandboxOperationRejected.from_error(error) from error
                _atomic_write(self._root, target, updated)
            digest = hashlib.sha256(updated.encode("utf-8")).hexdigest()
            return _write_result(normalized, digest)

        return await self._run_sync(operation)

    async def list_directory(self, path: str = ".") -> str:
        validate_request_size("list_directory", {"path": path})

        def operation() -> str:
            target, display, resource_key = self._read_target(
                path,
                directory=True,
            )
            try:
                candidates: Iterable[Path] = target.iterdir()
                if self._read_policy is not None:
                    candidates = (
                        entry
                        for entry in candidates
                        if self._policy_lists_entry(
                            entry,
                            display=display,
                            parent=target,
                            resource_key=resource_key,
                        )
                    )
                entries = heapq.nsmallest(
                    _MAX_ITEMS + 1,
                    candidates,
                    key=lambda item: item.name,
                )
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            visible = entries[:_MAX_ITEMS]
            incomplete = len(entries) > _MAX_ITEMS
            rows: list[str] = []
            for entry in visible:
                try:
                    info = entry.lstat()
                except OSError:
                    incomplete = True
                    continue
                if stat.S_ISLNK(info.st_mode):
                    if self._read_policy is not None:
                        continue
                    if not self._visible_read_path(
                        _child_display(display, entry, target),
                        resource_key=resource_key,
                        directory=False,
                    ):
                        continue
                    rows.append(entry.name + "@")
                elif stat.S_ISDIR(info.st_mode):
                    if not self._visible_read_path(
                        _child_display(display, entry, target),
                        resource_key=resource_key,
                        directory=True,
                    ):
                        continue
                    rows.append(entry.name + "/")
                elif stat.S_ISREG(info.st_mode):
                    if not self._visible_read_path(
                        _child_display(display, entry, target),
                        resource_key=resource_key,
                        directory=False,
                    ):
                        continue
                    rows.append(f"{entry.name}  ({info.st_size} bytes)")
            result = "\n".join(rows) if rows else "(empty directory)"
            if incomplete:
                return _bound_output_with_marker(result, "\n[truncated]")
            return _bound_output(result)

        return await self._run_sync(operation)

    def _policy_lists_entry(
        self,
        entry: Path,
        *,
        display: str,
        parent: Path,
        resource_key: str | None,
    ) -> bool:
        try:
            info = entry.lstat()
        except OSError:
            return False
        if stat.S_ISLNK(info.st_mode):
            return False
        is_directory = stat.S_ISDIR(info.st_mode)
        if not is_directory and not stat.S_ISREG(info.st_mode):
            return False
        return self._visible_read_path(
            _child_display(display, entry, parent),
            resource_key=resource_key,
            directory=is_directory,
        )

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        expression = _compile_pattern(pattern)
        if include_glob is not None and not isinstance(include_glob, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if include_glob is not None:
            _validate_glob(include_glob)
        validate_request_size(
            "search_files",
            {
                "pattern": pattern,
                "path": path,
                "include_glob": include_glob,
            },
        )

        def operation() -> str:
            target, display, resource_key = self._read_target(
                path,
                directory=True,
            )
            walk_state = _WalkState()
            if target.is_file():
                candidates: Iterable[Path] = (target,)
            elif target.is_dir():
                candidates = _walk_files(target, walk_state)
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

            rows: list[str] = []
            row_chars = 0
            incomplete = False
            for candidate in candidates:
                relative = _child_display(display, candidate, target)
                if not self._visible_read_path(
                    relative,
                    resource_key=resource_key,
                    directory=False,
                ):
                    continue
                if include_glob is not None and not fnmatch.fnmatchcase(
                    relative, include_glob
                ):
                    continue
                try:
                    matches, file_truncated = _search_file_lines(
                        candidate,
                        relative,
                        expression,
                        _MAX_ITEMS - len(rows),
                    )
                except AIError as error:
                    if error.code in {
                        ErrorCode.REQUEST_FIELD_INVALID,
                        ErrorCode.STORAGE_NOT_FOUND,
                        ErrorCode.STORAGE_UNAVAILABLE,
                        ErrorCode.STORAGE_CONFLICT,
                    }:
                        incomplete = True
                        continue
                    raise
                for row in matches:
                    separator_chars = 1 if rows else 0
                    if (
                        row_chars
                        + separator_chars
                        + len(row)
                        > _MAX_RENDERED_CHARS - len("\n[results incomplete]")
                    ):
                        incomplete = True
                        break
                    rows.append(row)
                    row_chars += separator_chars + len(row)
                if len(rows) >= _MAX_ITEMS or file_truncated:
                    incomplete = True
                if incomplete and (
                    len(rows) >= _MAX_ITEMS
                    or row_chars >= _MAX_RENDERED_CHARS - len("\n[results incomplete]")
                ):
                    break
            if walk_state.failed:
                incomplete = True
            result = "\n".join(rows) if rows else "No matches found."
            if incomplete:
                return _bound_output_with_marker(
                    result,
                    "\n[results incomplete]",
                )
            return _bound_output(result)

        return await self._run_sync(operation)

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        _validate_glob(pattern)
        validate_request_size("find_files", {"pattern": pattern, "path": path})

        def operation() -> str:
            directory, display, resource_key = self._read_target(
                path,
                directory=True,
            )
            if not directory.is_dir():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            matches: list[str] = []
            too_many = False
            walk_state = _WalkState()
            for target in _walk_files_and_directories(directory, walk_state):
                relative = _child_display(display, target, directory)
                if not self._visible_read_path(
                    relative,
                    resource_key=resource_key,
                    directory=target.is_dir(),
                ):
                    continue
                if target.is_symlink():
                    continue
                if fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase(
                    target.name, pattern
                ):
                    if len(matches) >= _MAX_ITEMS:
                        too_many = True
                        break
                    matches.append(relative)
            matches.sort()
            result = "\n".join(matches) if matches else "No matches found."
            if too_many or walk_state.failed:
                return _bound_output_with_marker(
                    result,
                    "\n[results incomplete]",
                )
            return _bound_output(result)

        return await self._run_sync(operation)

    async def create_directory(self, path: str) -> str:
        self._reject_read_only()
        try:
            normalized = _normalize_path(path)
            validate_request_size("create_directory", {"path": path})
        except AIError as error:
            raise SandboxOperationRejected.from_error(error) from error

        def operation() -> str:
            try:
                if _is_protected(self._workspace, normalized):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                target = self._directory_path(normalized, allow_missing=True)
                relative = _relative(self._root, target)
                if _is_protected(self._workspace, relative):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                if target.exists():
                    return _write_result(normalized, "directory", status="exists")
                _check_parent_chain(self._root, target.parent)
            except AIError as error:
                raise SandboxOperationRejected.from_error(error) from error
            with _file_lock(self._lock_root, relative):
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except FileNotFoundError as error:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
                except OSError as error:
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
                return _write_result(normalized, "directory", status="created")

        return await self._run_sync(operation)

    async def file_info(self, path: str) -> str:
        validate_request_size("file_info", {"path": path})

        def operation() -> str:
            target, display, _resource_key = self._read_target(
                path,
                directory=None,
            )
            self._authorize_read(
                display,
                resource_key=_resource_key,
                directory=target.is_dir(),
            )
            original = (
                self._root / _normalize_path(path)
                if self._read_policy is None
                else target
            )
            is_link = original.is_symlink()
            try:
                info = target.stat()
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
            parts = [
                f"path: {display}",
                f"type: {kind}",
                f"size: {info.st_size} bytes",
                f"mode: {stat.S_IMODE(info.st_mode):04o}",
            ]
            if stat.S_ISREG(info.st_mode):
                binary = _binary_file_size(target) is not None
                parts.append(f"binary: {str(binary).lower()}")
                if not binary:
                    digest, line_count, _visible, _truncated = _read_selected_lines(
                        target,
                        display,
                        0,
                        1,
                    )
                    parts.append(f"lines: {line_count}")
                    parts.append(f"hash: {digest}")
            if is_link:
                parts.append(f"symlink_target: {_relative(self._root, target.resolve())}")
            return _bound_output("\n".join(parts))

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
            raise SandboxOperationRejected.from_error(error) from error
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
                await _stop_process(process, force=True)
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
            raise SandboxOperationRejected.from_error(error) from error
        await self._ensure_open()
        async with self._process_lock:
            if len(self._processes) + self._starting_background >= 256:
                raise SandboxOperationRejected(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
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
            raise SandboxOperationRejected.from_error(error) from error
        process = await self._get_process(command_id)
        if process is None:
            raise SandboxOperationRejected(ErrorCode.REQUEST_FIELD_INVALID)
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
                await asyncio.shield(task)
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
        for wait in (self._wait_operations, self._wait_starting_tasks):
            try:
                await wait()
            except BaseException as error:
                cleanup_error = cleanup_error or error
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
        current = asyncio.current_task()
        while True:
            operations = tuple(
                task
                for task in self._operations
                if task is not current and not task.done()
            )
            if not operations:
                return
            await asyncio.gather(
                *(asyncio.shield(task) for task in operations),
                return_exceptions=True,
            )

    async def _wait_starting_tasks(self) -> None:
        current = asyncio.current_task()
        while True:
            tasks = tuple(
                task
                for task in self._starting_tasks
                if task is not current and not task.done()
            )
            if not tasks:
                return
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )

    async def _ensure_open(self) -> None:
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
            raise SandboxOperationRejected(ErrorCode.AUTHORIZATION_DENIED)

    def _authorize_read(
        self,
        path: str,
        *,
        resource_key: str | None = None,
        directory: bool | None = False,
    ) -> None:
        if self._read_policy is None:
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
        if self._read_policy is None:
            return True
        try:
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
            self._read_policy is not None
            and isinstance(path, str)
            and Path(path).is_absolute()
        ):
            candidate = Path(path)
            resource_matches = tuple(
                (key, source)
                for key, source in self._resources.items()
                if _inside_or_equal(source, candidate)
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

    async def _run_sync(self, operation: Callable[[], str]) -> str:
        current = asyncio.current_task()
        async with self._state_lock:
            if self._state != "OPEN":
                code = (
                    ErrorCode.SANDBOX_SESSION_LOST
                    if self._state == "LOST"
                    else ErrorCode.SANDBOX_SESSION_CLOSED
                )
                raise AIError(code)
            if current is not None:
                self._operations.add(current)
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except BaseException as error:
                _logger.exception(
                    "cancelled local sandbox operation completed with an error",
                    exc_info=error,
                )
            raise
        finally:
            if current is not None:
                self._operations.discard(current)

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

    def _file_path(self, path: str, *, write: bool) -> Path:
        if write and _is_protected(self._workspace, path):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        candidate = self._resolve_path(path, allow_missing=write)
        resolved = candidate.resolve(strict=False)
        if write and _is_protected(self._workspace, _relative(self._root, resolved)):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if write and not candidate.parent.is_dir():
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if candidate.exists():
            info = candidate.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif not write:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return candidate

    def _directory_path(self, path: str, *, allow_missing: bool = False) -> Path:
        candidate = self._root / path
        if candidate.is_symlink():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        candidate = self._resolve_path(path, allow_missing=allow_missing)
        if candidate.exists() and not candidate.is_dir():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return candidate

    async def _start_process(self, command: str) -> "_ProcessState":
        current = asyncio.current_task()
        async with self._process_lock:
            if self._state != "OPEN":
                raise AIError(_session_state_error(self._state))
            if current is not None:
                self._starting_tasks.add(current)
        try:
            return await self._spawn_process(command)
        finally:
            if current is not None:
                self._starting_tasks.discard(current)

    async def _spawn_process(self, command: str) -> "_ProcessState":
        await self._ensure_open()
        environment = dict(self._environment)
        job: _WindowsJob | None = None
        if os.name == "nt":
            job = _WindowsJob.create()
        if os.name == "nt":
            executable = self._environment.get("COMSPEC", "cmd.exe")
        else:
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
            if os.name == "nt":
                return await asyncio.create_subprocess_exec(
                    executable,
                    "/d",
                    "/c",
                    command,
                    **process_kwargs,
                )
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
                process = await asyncio.shield(process_task)
            except BaseException:
                if job is not None:
                    job.close()
                raise
            process_key = id(process)
            self._pending_processes[process_key] = (process, job)
            try:
                await _terminate_unregistered_process(process, job)
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
                try:
                    await stdin.wait_closed()
                except (BrokenPipeError, ConnectionError, OSError):
                    pass

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
                await asyncio.shield(task)
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
    args: Sequence[str | SandboxResourcePath],
    resources: Sequence[SandboxResource],
) -> tuple[str, ...]:
    by_id = {resource.id: resource for resource in resources}
    result: list[str] = []
    for argument in args:
        if isinstance(argument, str):
            if "\x00" in argument:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            result.append(argument)
            continue
        if not isinstance(argument, SandboxResourcePath):
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
    resources: tuple[SandboxResource, ...],
) -> tuple[SandboxResource, ...]:
    if not isinstance(resources, tuple):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    seen: set[str] = set()
    values: list[SandboxResource] = []
    root = workspace.root
    storage_root = workspace.storage_root
    for resource in resources:
        if not isinstance(resource, SandboxResource) or resource.id in seen:
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


def _walk_files(root: Path, state: _WalkState) -> Iterable[Path]:
    def onerror(error: OSError) -> None:
        del error
        state.failed = True

    for current, directories, files in os.walk(
        root,
        followlinks=False,
        onerror=onerror,
    ):
        directories[:] = sorted(
            name for name in directories if not (Path(current) / name).is_symlink()
        )
        for name in sorted(files):
            target = Path(current) / name
            if target.is_symlink():
                continue
            try:
                info = target.lstat()
            except OSError:
                state.failed = True
                continue
            if stat.S_ISREG(info.st_mode):
                yield target


def _walk_files_and_directories(
    root: Path,
    state: _WalkState,
) -> Iterable[Path]:
    def onerror(error: OSError) -> None:
        del error
        state.failed = True

    yield root
    for current, directories, files in os.walk(
        root,
        followlinks=False,
        onerror=onerror,
    ):
        directories[:] = sorted(
            name for name in directories if not (Path(current) / name).is_symlink()
        )
        for name in sorted(directories):
            yield Path(current) / name
        for name in sorted(files):
            target = Path(current) / name
            if target.is_symlink():
                continue
            try:
                info = target.lstat()
            except OSError:
                state.failed = True
                continue
            if stat.S_ISREG(info.st_mode):
                yield target


def _read_selected_lines(
    path: Path,
    logical: str,
    offset: int,
    limit: int,
) -> tuple[str, int, list[str], bool]:
    try:
        digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        selected: list[str] = []
        selected_chars = 0
        selected_truncated = False
        line_count = 0
        line_parts: list[str] = []
        line_chars = 0
        line_truncated = False
        line_started = False
        pending_cr = False

        def append_text(value: str) -> None:
            nonlocal line_chars, line_truncated, selected_truncated
            if line_count < offset or line_count >= offset + limit:
                return
            if line_truncated:
                return
            remaining = min(
                _MAX_RENDERED_CHARS - line_chars,
                _MAX_RENDERED_CHARS - 256 - selected_chars,
            )
            if len(value) > remaining:
                if remaining:
                    line_parts.append(value[:remaining])
                    line_chars += remaining
                line_truncated = True
                selected_truncated = True
                return
            line_parts.append(value)
            line_chars += len(value)

        def finish_line(ended: bool) -> None:
            nonlocal line_count, line_parts, line_chars, line_truncated
            nonlocal selected_chars, selected_truncated
            if offset <= line_count < offset + limit:
                value = "".join(line_parts)
                if line_truncated:
                    value += "...[line truncated]"
                    if ended:
                        value += "\n"
                remaining = _MAX_RENDERED_CHARS - 256 - selected_chars
                if remaining <= 0:
                    selected_truncated = True
                elif len(value) <= remaining:
                    selected.append(value)
                    selected_chars += len(value)
                else:
                    marker = "...[output truncated]"
                    if remaining > len(marker):
                        selected.append(value[: remaining - len(marker)] + marker)
                        selected_chars += remaining
                    selected_truncated = True
            line_count += 1
            line_parts = []
            line_chars = 0
            line_truncated = False

        def consume(value: str) -> None:
            nonlocal line_started, pending_cr
            for character in value:
                if pending_cr:
                    if character == "\n":
                        append_text(character)
                        finish_line(ended=True)
                        pending_cr = False
                        line_started = False
                        continue
                    finish_line(ended=True)
                    pending_cr = False
                    line_started = False
                if character == "\r":
                    line_started = True
                    append_text(character)
                    pending_cr = True
                elif character in "\n\v\f\x1c\x1d\x1e\x85\u2028\u2029":
                    line_started = True
                    append_text(character)
                    finish_line(ended=True)
                    line_started = False
                else:
                    if character == "\x00":
                        character = "\ufffd"
                    line_started = True
                    append_text(character)

        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            for chunk in stream:
                digest.update(chunk)
                try:
                    consume(decoder.decode(chunk, final=False))
                except UnicodeDecodeError as error:
                    raise AIError(
                        ErrorCode.REQUEST_FIELD_INVALID,
                        safe_details={"path": logical, "reason": "invalid_utf8"},
                    ) from error
            try:
                consume(decoder.decode(b"", final=True))
            except UnicodeDecodeError as error:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"path": logical, "reason": "invalid_utf8"},
                ) from error
            if pending_cr:
                finish_line(ended=True)
            elif line_started:
                finish_line(ended=False)
            if _file_identity(stream.fileno()) != identity or _path_identity(path) != identity:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
    except FileNotFoundError as error:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
    return digest.hexdigest(), line_count, selected, selected_truncated


def _read_text_with_digest(path: Path, logical: str) -> tuple[str, str]:
    try:
        digest = hashlib.sha256()
        value = io.StringIO()
        decoder = codecs.getincrementaldecoder("utf-8")()
        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                decoded = decoder.decode(chunk, final=False)
                if "\x00" in decoded:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                value.write(decoded)
            decoded = decoder.decode(b"", final=True)
            if "\x00" in decoded:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            value.write(decoded)
            if _file_identity(stream.fileno()) != identity or _path_identity(path) != identity:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return value.getvalue(), digest.hexdigest()
    except UnicodeDecodeError as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"path": logical, "reason": "invalid_utf8"},
        ) from error
    except FileNotFoundError as error:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _read_text(path: Path, logical: str) -> str:
    return _read_text_with_digest(path, logical)[0]


def _binary_file_size(path: Path) -> int | None:
    try:
        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            info = os.fstat(stream.fileno())
            sample = stream.read(8192)
            if (
                _file_identity(stream.fileno()) != identity
                or _path_identity(path) != identity
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return info.st_size if b"\x00" in sample else None
    except AIError:
        raise
    except FileNotFoundError as error:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _search_file_lines(
    path: Path,
    logical: str,
    expression: re.Pattern[str],
    limit: int,
) -> tuple[list[str], bool]:
    if limit <= 0:
        return [], True
    try:
        with path.open("rb") as probe:
            identity = _file_identity(probe.fileno())
            if b"\x00" in probe.read(8192):
                return [], False
        matches: list[str] = []
        truncated = False
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline=None,
        ) as stream:
            if _file_identity(stream.fileno()) != identity:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            for line_number, line in enumerate(stream, 1):
                value = line.rstrip("\r\n")
                if not expression.search(value):
                    continue
                if len(matches) >= limit:
                    truncated = True
                    break
                matches.append(f"{logical}:{line_number}:{value}")
            if (
                _file_identity(stream.fileno()) != identity
                or _path_identity(path) != identity
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return matches, truncated
    except AIError:
        raise
    except FileNotFoundError as error:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _read_optional_text(path: Path, logical: str) -> str | None:
    if not path.exists():
        return None
    return _read_text(path, logical)


def _read_optional_hash(path: Path, logical: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            for chunk in stream:
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


def _check_expected_hash(current: str | None, expected: str | None) -> None:
    if expected is None:
        return
    if current is None:
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    actual = hashlib.sha256(current.encode("utf-8")).hexdigest()
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


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
            stream.write(content.encode("utf-8"))
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


def _render_read(
    logical: str,
    digest: str,
    line_count: int,
    offset: int,
    visible: list[str],
    selected_truncated: bool,
) -> str:
    if line_count == 0:
        return f"path: {logical}\nhash: {digest}\nlines: 0/0"
    end = offset + len(visible)
    body = "".join(
        f"{offset + index + 1}: {line}" for index, line in enumerate(visible)
    )
    if selected_truncated:
        body += "\n...[output truncated]"
    if visible and len(body) > _MAX_RENDERED_CHARS - 180:
        body = body[: _MAX_RENDERED_CHARS - 180] + "...[line truncated]"
    return (
        f"path: {logical}\n"
        f"hash: {digest}\n"
        f"lines: {offset + 1}-{end}/{line_count}\n"
        + body
    )


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




__all__ = ["LocalSandbox"]
