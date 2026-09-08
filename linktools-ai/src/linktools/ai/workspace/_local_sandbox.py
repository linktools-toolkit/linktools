#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local workspace backend.

The local backend provides the workspace contract and lifecycle guarantees; it
is deliberately not a security sandbox.  Agent code still only receives the
logical file and command operations exposed by ``SandboxSession``.
"""

import asyncio
import codecs
import ctypes
import errno
import fnmatch
import heapq
import hashlib
import io
import os
import re
import shlex
import signal
import stat
import sys
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, PIPE

from filelock import FileLock
from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._sandbox import (
    SandboxResource,
    SandboxSession,
    normalize_workspace_path,
)
from ._sandbox_protocol import validate_request_size

_logger = environ.get_logger("ai.workspace.local_sandbox")
_MAX_OUTPUT_CHARS = 50_000
_MAX_RENDERED_CHARS = 50_000
_MAX_CONTENT_CHARS = 50_000
_MAX_SEARCH_CONTENT_CHARS = 65_536
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
    """Open local sessions rooted at the caller-provided workspace."""

    def __init__(self) -> None:
        pass

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        normalized_root = _normalize_root(root)
        normalized_resources = _validate_resources(normalized_root, resources)
        _logger.debug(
            "opening local sandbox session: root=%s resources=%s",
            normalized_root,
            tuple(resource.key for resource in normalized_resources),
        )
        lock_root = normalized_root / ".linktools" / "locks"
        _prepare_lock_root(lock_root)
        return _LocalSandboxSession(
            normalized_root,
            normalized_resources,
            lock_root=lock_root,
        )


class _WindowsJob:
    """Own one Windows process tree with kill-on-close semantics."""

    _KILL_ON_CLOSE = 0x00002000
    _PROCESS_ACCESS = 0x0001 | 0x0100 | 0x0800 | 0x1000

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._closed = False

    @classmethod
    def create(cls) -> "_WindowsJob":
        if os.name != "nt":
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        job = cls(int(handle))
        try:
            job._configure()
        except BaseException:
            job.close()
            raise
        return job

    def _configure(self) -> None:
        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [("values", ctypes.c_uint64 * 6)]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = self._KILL_ON_CLOSE
        kernel32 = ctypes.windll.kernel32
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        if not kernel32.SetInformationJobObject(
            self._handle,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        kernel32.SetHandleInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        if not kernel32.SetHandleInformation(self._handle, 1, 0):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)

    def assign_and_resume(self, process_id: int) -> None:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        process = kernel32.OpenProcess(self._PROCESS_ACCESS, False, process_id)
        if not process:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        try:
            kernel32.AssignProcessToJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            kernel32.AssignProcessToJobObject.restype = ctypes.c_int
            if not kernel32.AssignProcessToJobObject(self._handle, process):
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
            ntdll = ctypes.windll.ntdll
            ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
            ntdll.NtResumeProcess.restype = ctypes.c_long
            if ntdll.NtResumeProcess(process) != 0:
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        finally:
            kernel32.CloseHandle(process)

    def terminate(self) -> None:
        if self._closed:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.TerminateJobObject.restype = ctypes.c_int
        if not kernel32.TerminateJobObject(self._handle, 1):
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not ctypes.windll.kernel32.CloseHandle(self._handle):
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)


class _LocalSandboxSession:
    def __init__(
        self,
        root: Path,
        resources: tuple[SandboxResource, ...],
        *,
        lock_root: Path | None = None,
    ) -> None:
        self._root = root
        self._resources = {resource.key: resource.source.resolve() for resource in resources}
        self._lock_root = lock_root or root / ".linktools" / "locks"
        self._environment = _command_environment()
        self._state = "OPEN"
        self._state_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._processes: dict[str, _ProcessState] = {}
        self._pending_processes: dict[
            int, tuple[asyncio.subprocess.Process, _WindowsJob | None]
        ] = {}
        self._operations: set[asyncio.Task[object]] = set()
        self._starting_tasks: set[asyncio.Task[object]] = set()
        self._starting_background = 0
        self._process_lock = asyncio.Lock()

    def resource_path(self, key: str) -> str:
        self._ensure_open_sync()
        if not isinstance(key, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = self._resources.get(key)
        if source is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
        return frozenset(values)

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        normalized = _normalize_path(path)
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
            target = self._file_path(normalized, write=False)
            digest, line_count, visible, selected_truncated = _read_selected_lines(
                target,
                normalized,
                offset,
                selected_limit,
            )
            if offset >= line_count and line_count:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            rendered = _render_read(
                normalized,
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
        normalized = _normalize_path(path)
        expected = _validate_expected_hash(expected_hash)
        validate_request_size(
            "write_file",
            {"path": path, "content": content, "expected_hash": expected_hash},
        )
        _validate_text(content)

        def operation() -> str:
            target = self._file_path(normalized, write=True)
            with _file_lock(self._lock_root, _relative(self._root, target)):
                if expected is not None:
                    current_hash = _read_optional_hash(target, normalized)
                    _check_expected_digest(current_hash, expected)
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

        def operation() -> str:
            target = self._file_path(normalized, write=True)
            with _file_lock(self._lock_root, _relative(self._root, target)):
                current = _read_optional_text(target, normalized)
                _check_expected_hash(current, expected)
                if current is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND)
                if current.count(old_text) != 1:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                updated = current.replace(old_text, new_text)
                _validate_text(updated)
                _atomic_write(self._root, target, updated)
            digest = hashlib.sha256(updated.encode("utf-8")).hexdigest()
            return _write_result(normalized, digest)

        return await self._run_sync(operation)

    async def list_directory(self, path: str = ".") -> str:
        normalized = _normalize_path(path)
        validate_request_size("list_directory", {"path": path})

        def operation() -> str:
            target = self._directory_path(normalized)
            try:
                entries = heapq.nsmallest(
                    _MAX_ITEMS + 1,
                    target.iterdir(),
                    key=lambda item: item.name,
                )
            except FileNotFoundError as error:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
            except OSError as error:
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            if len(entries) > _MAX_ITEMS:
                visible = entries[:_MAX_ITEMS]
                truncated = True
            else:
                visible = entries
                truncated = False
            rows = []
            for entry in visible:
                if entry.is_symlink():
                    marker = "@"
                elif entry.is_dir():
                    marker = "/"
                else:
                    marker = ""
                rows.append(entry.name + marker)
            if truncated:
                rows.append("[truncated]")
            return _bound_output("\n".join(rows))

        return await self._run_sync(operation)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        normalized = _normalize_path(path)
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
            directory = self._directory_path(normalized)
            rows: list[str] = []
            row_chars = 0
            truncated = False
            walk_state = _WalkState()
            for target in _walk_files(directory, walk_state):
                relative = _relative(self._root, target)
                if include_glob is not None and not fnmatch.fnmatchcase(
                    relative, include_glob
                ):
                    continue
                try:
                    content, content_truncated = _read_search_text(target, relative)
                    truncated = truncated or content_truncated
                except AIError as error:
                    if error.code in {
                        ErrorCode.REQUEST_FIELD_INVALID,
                        ErrorCode.STORAGE_NOT_FOUND,
                        ErrorCode.STORAGE_UNAVAILABLE,
                    }:
                        truncated = True
                        continue
                    raise
                for line_number, line in enumerate(content.splitlines(), 1):
                    if expression.search(line):
                        if len(rows) >= _MAX_ITEMS:
                            truncated = True
                            break
                        row = f"{relative}:{line_number}:{line}"
                        separator_chars = 1 if rows else 0
                        if (
                            row_chars
                            + separator_chars
                            + len(row)
                            > _MAX_RENDERED_CHARS - len("\n[results incomplete]")
                        ):
                            truncated = True
                            break
                        rows.append(row)
                        row_chars += separator_chars + len(row)
                if truncated:
                    break
            if walk_state.failed:
                truncated = True
            result = "\n".join(rows)
            if truncated:
                return _bound_output_with_marker(
                    result,
                    "\n[results incomplete]",
                )
            return _bound_output(result)

        return await self._run_sync(operation)

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        normalized = _normalize_path(path)
        _validate_glob(pattern)
        validate_request_size("find_files", {"pattern": pattern, "path": path})

        def operation() -> str:
            directory = self._directory_path(normalized)
            matches: list[str] = []
            too_many = False
            walk_state = _WalkState()
            for target in _walk_files_and_directories(directory, walk_state):
                relative = _relative(self._root, target)
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
            result = "\n".join(matches)
            if too_many or walk_state.failed:
                return _bound_output_with_marker(
                    result,
                    "\n[results incomplete]",
                )
            return _bound_output(result)

        return await self._run_sync(operation)

    async def create_directory(self, path: str) -> str:
        normalized = _normalize_path(path)
        validate_request_size("create_directory", {"path": path})

        def operation() -> str:
            if _is_protected(normalized):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            target = self._directory_path(normalized, allow_missing=True)
            relative = _relative(self._root, target)
            with _file_lock(self._lock_root, relative):
                if _is_protected(relative):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                if target.exists():
                    return _write_result(normalized, "directory", status="exists")
                _check_parent_chain(self._root, target.parent)
                try:
                    target.mkdir(parents=True, exist_ok=True)
                except FileNotFoundError as error:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
                except OSError as error:
                    raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
                return _write_result(normalized, "directory", status="created")

        return await self._run_sync(operation)

    async def file_info(self, path: str) -> str:
        normalized = _normalize_path(path)
        validate_request_size("file_info", {"path": path})

        def operation() -> str:
            target = self._resolve_path(normalized, allow_missing=False)
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
            return _bound_output(
                f"path: {normalized}\n"
                f"type: {kind}\n"
                f"size: {info.st_size}\n"
                f"mode: {stat.S_IMODE(info.st_mode):04o}"
            )

        return await self._run_sync(operation)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        validate_request_size(
            "run_command",
            {"command": command, "timeout_seconds": timeout_seconds},
        )
        _validate_command_input(command)
        _validate_command(command)
        timeout = _command_timeout(timeout_seconds)
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
        validate_request_size("start_command", {"command": command})
        _validate_command_input(command)
        _validate_command(command)
        await self._ensure_open()
        async with self._process_lock:
            if len(self._processes) + self._starting_background >= 256:
                raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
            self._starting_background += 1
        try:
            process = await self._start_process(command)
        finally:
            async with self._process_lock:
                self._starting_background -= 1
        return _bound_output(f"command_id: {process.command_id}\nstatus: running")

    async def check_command(self, command_id: str) -> str:
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
        validate_request_size("stop_command", {"command_id": command_id})
        process = await self._get_process(command_id)
        if process is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
            pending_processes = tuple(self._pending_processes.items())
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

    def _resolve_path(self, path: str, *, allow_missing: bool) -> Path:
        self._ensure_open_sync()
        candidate = self._root / path
        _check_parent_chain(self._root, candidate.parent)
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
        if not _inside(self._root, resolved):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if not allow_missing and not resolved.exists():
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return resolved if candidate.is_symlink() else candidate

    def _file_path(self, path: str, *, write: bool) -> Path:
        if write and _is_protected(path):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        candidate = self._resolve_path(path, allow_missing=write)
        resolved = candidate.resolve(strict=False)
        if write and _is_protected(_relative(self._root, resolved)):
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
        state.stdout_reader_task = asyncio.create_task(
            _read_process_output(state, "stdout"),
            name=f"sandbox-stdout-{command_id}",
        )
        state.stderr_reader_task = asyncio.create_task(
            _read_process_output(state, "stderr"),
            name=f"sandbox-stderr-{command_id}",
        )
        state.wait_task = asyncio.create_task(
            _wait_process(state),
            name=f"sandbox-wait-{command_id}",
        )
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


async def _stop_process(state: "_ProcessState", *, force: bool) -> None:
    async with state.stop_lock:
        if state.final_status == "stopped" and state.wait_task.done():
            return
        requested_stop = state.process.returncode is None
        cleanup_error: Exception | None = None
        if force and state.job is not None:
            try:
                state.job.terminate()
            except Exception as error:
                cleanup_error = error
        if state.process.returncode is None:
            try:
                _signal_process(
                    state.process,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            except Exception as error:
                cleanup_error = cleanup_error or error
        try:
            await asyncio.wait_for(asyncio.shield(state.wait_task), 5.0)
        except asyncio.TimeoutError:
            try:
                _signal_process_group_if_owned(state, signal.SIGKILL)
            except Exception as error:
                cleanup_error = cleanup_error or error
            try:
                await asyncio.wait_for(asyncio.shield(state.wait_task), 5.0)
            except asyncio.TimeoutError as error:
                cleanup_error = cleanup_error or error
            except Exception as error:
                cleanup_error = cleanup_error or error
        except Exception as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            if isinstance(cleanup_error, AIError):
                raise cleanup_error
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from cleanup_error
        if requested_stop:
            state.final_status = "stopped"


async def _stop_process_state(state: "_ProcessState", *, force: bool) -> None:
    await _stop_process(state, force=force)


async def _terminate_unregistered_process(
    process: asyncio.subprocess.Process,
    job: _WindowsJob | None,
) -> None:
    cleanup_error: BaseException | None = None
    try:
        if job is not None:
            try:
                job.terminate()
            except BaseException as error:
                cleanup_error = error
        if process.returncode is None:
            if os.name == "posix":
                _signal_process_group(process.pid, signal.SIGKILL)
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), 5.0)
        except asyncio.TimeoutError as error:
            cleanup_error = error
    finally:
        if job is not None:
            try:
                job.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from cleanup_error


class _ProcessState:
    def __init__(
        self,
        command_id: str,
        command: str,
        process: asyncio.subprocess.Process,
        *,
        job: _WindowsJob | None = None,
    ) -> None:
        self.command_id = command_id
        self.command = command
        self.process = process
        self.stdout: deque[str] = deque()
        self.stderr: deque[str] = deque()
        self.stdout_chars = 0
        self.stderr_chars = 0
        self.output_incomplete = False
        self.stdout_decoder = codecs.getincrementaldecoder("utf-8")(
            errors="replace"
        )
        self.stderr_decoder = codecs.getincrementaldecoder("utf-8")(
            errors="replace"
        )
        self.final_status: str | None = None
        current = asyncio.current_task()
        if current is None:
            raise RuntimeError("process state requires an asyncio task")
        self.stdout_reader_task: asyncio.Task[None] = current  # replaced below
        self.stderr_reader_task: asyncio.Task[None] = current  # replaced below
        self.wait_task: asyncio.Task[None] = current  # replaced below
        self.stop_lock = asyncio.Lock()
        self.job = job
        try:
            if sys.platform.startswith("linux"):
                fields = _proc_stat_fields(process.pid)
                self.process_start_time = fields[19]
                self.process_session_id = fields[3]
            else:
                self.process_start_time = None
                self.process_session_id = None
        except OSError:
            self.process_start_time = None
            self.process_session_id = None


async def _read_process_output(state: _ProcessState, channel: str) -> None:
    if channel == "stdout":
        stream = state.process.stdout
        decoder = state.stdout_decoder
    elif channel == "stderr":
        stream = state.process.stderr
        decoder = state.stderr_decoder
    else:
        raise ValueError("unknown process output channel")
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                text = decoder.decode(b"", final=True)
                if "\ufffd" in text:
                    state.output_incomplete = True
                _append_process_output(state, channel, text)
                return
            text = decoder.decode(chunk)
            if "\ufffd" in text:
                state.output_incomplete = True
            _append_process_output(state, channel, text)
    except asyncio.CancelledError:
        raise
    except OSError:
        state.output_incomplete = True


def _append_process_output(state: _ProcessState, channel: str, text: str) -> None:
    if not text:
        return
    if channel == "stdout":
        output = state.stdout
        state.stdout_chars += len(text)
        chars = state.stdout_chars
    elif channel == "stderr":
        output = state.stderr
        state.stderr_chars += len(text)
        chars = state.stderr_chars
    else:
        raise ValueError("unknown process output channel")
    output.append(text)
    while chars > _MAX_OUTPUT_CHARS and output:
        removed = output.popleft()
        excess = chars - _MAX_OUTPUT_CHARS
        if len(removed) <= excess:
            chars -= len(removed)
        else:
            output.appendleft(removed[excess:])
            chars -= excess
        state.output_incomplete = True
    if channel == "stdout":
        state.stdout_chars = chars
    else:
        state.stderr_chars = chars


async def _wait_process(state: _ProcessState) -> None:
    cleanup_error: BaseException | None = None
    try:
        group_cleanup_needed = await _observe_process_exit(state)
        if state.job is not None:
            try:
                state.job.terminate()
            except BaseException as error:
                cleanup_error = error
                state.output_incomplete = True
        if group_cleanup_needed:
            _signal_process_group_if_owned(state, signal.SIGTERM)
            await _wait_for_process_group(state, signal.SIGKILL)
        reader_tasks = (state.stdout_reader_task, state.stderr_reader_task)
        pending_readers = tuple(task for task in reader_tasks if not task.done())
        if pending_readers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(asyncio.shield(task) for task in pending_readers)
                    ),
                    2.0,
                )
            except asyncio.TimeoutError:
                state.output_incomplete = True
                _signal_process_group_if_owned(state, signal.SIGKILL)
                for task in pending_readers:
                    task.cancel()
                await asyncio.gather(*pending_readers, return_exceptions=True)
            except OSError:
                state.output_incomplete = True
        await state.process.wait()
        if state.final_status is None:
            state.final_status = "exited"
    except BaseException as error:
        cleanup_error = cleanup_error or error
    finally:
        if state.job is not None:
            try:
                state.job.close()
            except BaseException as error:
                cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        if isinstance(cleanup_error, AIError):
            raise cleanup_error
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from cleanup_error


async def _wait_for_process_group(
    state: _ProcessState,
    fallback_signal: signal.Signals,
) -> None:
    if not _has_owned_process_group_members(state):
        return
    deadline = asyncio.get_running_loop().time() + 5.0
    while _has_owned_process_group_members(state):
        if asyncio.get_running_loop().time() >= deadline:
            _signal_process_group_if_owned(state, fallback_signal)
            break
        await asyncio.sleep(0.02)
    deadline = asyncio.get_running_loop().time() + 5.0
    while _has_owned_process_group_members(state):
        if asyncio.get_running_loop().time() >= deadline:
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
        await asyncio.sleep(0.02)


async def _observe_process_exit(state: _ProcessState) -> bool:
    """Observe a POSIX child without reaping it before group cleanup."""
    process = state.process
    if os.name != "posix":
        await process.wait()
        return False
    if process.returncode is not None:
        await process.wait()
        return _process_group_is_owned(state)
    waitid = getattr(os, "waitid", None)
    if waitid is None or not hasattr(os, "P_PID"):
        await process.wait()
        return False
    while True:
        if process.returncode is not None:
            if not sys.platform.startswith("linux"):
                return False
            try:
                _proc_start_time(process.pid)
            except OSError:
                return _has_owned_process_group_members(state)
            return True
        try:
            result = waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError:
            return False
        except OSError as error:
            if error.errno in {errno.ECHILD, errno.EINVAL}:
                return False
            raise
        if result is not None and result.si_pid == process.pid:
            return True
        await asyncio.sleep(0.02)
    return False


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
    root: Path,
    resources: tuple[SandboxResource, ...],
) -> tuple[SandboxResource, ...]:
    if not isinstance(resources, tuple):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    seen: set[str] = set()
    values: list[SandboxResource] = []
    for resource in resources:
        if not isinstance(resource, SandboxResource) or resource.key in seen:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source = resource.source
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        if (
            not resolved.is_dir()
            or source.is_symlink()
            or resolved == root
            or _inside(resolved, root)
            or resolved == root / ".linktools"
            or _inside(resolved, root / ".linktools")
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        seen.add(resource.key)
        values.append(resource)
    return tuple(values)


def _normalize_path(path: str) -> str:
    value = normalize_workspace_path(path)
    if os.name == "nt":
        _validate_windows_path(value)
    return value


def _inside(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
    except ValueError:
        return False
    return True


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


def _is_protected(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if (
        normalized == ".linktools"
        or normalized.startswith(".linktools/")
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
        decoder = codecs.getincrementaldecoder("utf-8")()
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
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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


def _read_search_text(path: Path, logical: str) -> tuple[str, bool]:
    try:
        decoder = codecs.getincrementaldecoder("utf-8")()
        chunks: list[str] = []
        characters = 0
        truncated = False
        with path.open("rb") as stream:
            identity = _file_identity(stream.fileno())
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    try:
                        decoded = decoder.decode(b"", final=True)
                    except UnicodeDecodeError as error:
                        raise AIError(
                            ErrorCode.REQUEST_FIELD_INVALID,
                            safe_details={
                                "path": logical,
                                "reason": "invalid_utf8",
                            },
                        ) from error
                    if "\x00" in decoded:
                        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                    if decoded:
                        remaining = _MAX_SEARCH_CONTENT_CHARS - characters
                        if len(decoded) > remaining:
                            chunks.append(decoded[:remaining])
                            characters = _MAX_SEARCH_CONTENT_CHARS
                            truncated = True
                        else:
                            chunks.append(decoded)
                            characters += len(decoded)
                    break
                try:
                    decoded = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as error:
                    raise AIError(
                        ErrorCode.REQUEST_FIELD_INVALID,
                        safe_details={
                            "path": logical,
                            "reason": "invalid_utf8",
                        },
                    ) from error
                if "\x00" in decoded:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                if characters < _MAX_SEARCH_CONTENT_CHARS:
                    remaining = _MAX_SEARCH_CONTENT_CHARS - characters
                    if len(decoded) > remaining:
                        chunks.append(decoded[:remaining])
                        characters = _MAX_SEARCH_CONTENT_CHARS
                        truncated = True
                        break
                    elif decoded:
                        chunks.append(decoded)
                        characters += len(decoded)
                if characters == _MAX_SEARCH_CONTENT_CHARS:
                    probe = stream.read(1)
                    if probe:
                        truncated = True
                    else:
                        try:
                            decoder.decode(b"", final=True)
                        except UnicodeDecodeError:
                            truncated = True
                    break
            if _file_identity(stream.fileno()) != identity or _path_identity(path) != identity:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        return "".join(chunks), truncated
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
        target = parent / f".linktools-{uuid.uuid4().hex}"
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


def _signal_process(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix":
        _signal_process_group(process.pid, sig)
        return
    try:
        process.send_signal(sig)
    except ProcessLookupError:
        pass


def _signal_process_group(process_id: int, sig: signal.Signals) -> None:
    if os.name != "posix":
        return
    try:
        os.killpg(process_id, sig)
    except ProcessLookupError:
        return
    except OSError as error:
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error


def _signal_process_group_if_owned(
    state: _ProcessState,
    sig: signal.Signals,
) -> None:
    if os.name != "posix" or not _process_group_is_owned(state):
        return
    _signal_process_group(state.process.pid, sig)


def _process_group_is_owned(state: _ProcessState) -> bool:
    process_id = state.process.pid
    if sys.platform.startswith("linux"):
        try:
            if state.process_start_time is not None:
                if _proc_start_time(process_id) != state.process_start_time:
                    return False
                return os.getpgid(process_id) == process_id
        except OSError:
            pass
        return _has_owned_process_group_members(state)
    elif state.process.returncode is not None:
        return False
    try:
        return os.getpgid(process_id) == process_id
    except OSError:
        return False


def _proc_start_time(process_id: int) -> int:
    return _proc_stat_fields(process_id)[19]


def _proc_stat_fields(process_id: int) -> list[int | str]:
    value = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
    _, remainder = value.rsplit(")", 1)
    fields = remainder.split()
    if len(fields) <= 19:
        raise OSError("/proc stat is incomplete")
    return [fields[0], *(_parse_proc_stat_number(item) for item in fields[1:])]


def _parse_proc_stat_number(value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise OSError("/proc stat contains invalid data") from error


def _has_owned_process_group_members(state: _ProcessState) -> bool:
    session_id = state.process_session_id
    if session_id is None:
        return False
    process_id = state.process.pid
    if state.process_start_time is not None:
        try:
            if _proc_start_time(process_id) != state.process_start_time:
                return False
        except OSError:
            pass
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as error:
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            fields = _proc_stat_fields(int(entry.name))
        except OSError:
            continue
        if (
            int(entry.name) != process_id
            and fields[0] != "Z"
            and fields[2] == process_id
            and fields[3] == session_id
        ):
            return True
    return False


def _command_result(state: _ProcessState, *, status: str | None = None) -> str:
    actual_status = status or (
        state.final_status
        or ("running" if not state.wait_task.done() else "exited")
    )
    code = state.process.returncode
    header = (
        f"command_id: {state.command_id}\n"
        f"status: {actual_status}\n"
        f"exit_code: {'' if code is None else code}"
    )
    return _render_command_output(
        header,
        "".join(state.stdout),
        "".join(state.stderr),
        incomplete=state.output_incomplete,
    )


def _render_command_output(
    header: str,
    stdout: str,
    stderr: str,
    *,
    incomplete: bool,
) -> str:
    channels = [
        ("[stdout]\n", stdout),
        ("[stderr]\n", stderr),
    ]
    channels = [(label, value) for label, value in channels if value]
    body = "\n\n".join(label + value for label, value in channels)
    full = header if not body else f"{header}\n{body}"
    truncated = incomplete or len(full) > _MAX_RENDERED_CHARS
    if not truncated:
        return full

    suffix = "\n[output incomplete]"
    if not channels:
        return header + suffix
    fixed = (
        len(header)
        + 1
        + sum(len(label) for label, _value in channels)
        + 2 * (len(channels) - 1)
        + len(suffix)
    )
    available = max(0, _MAX_RENDERED_CHARS - fixed)
    if len(channels) == 1:
        budgets = [min(len(channels[0][1]), available)]
    else:
        first = min(len(channels[0][1]), available // 2)
        second = min(len(channels[1][1]), available - first)
        remaining = available - first - second
        if remaining and first < len(channels[0][1]):
            extra = min(remaining, len(channels[0][1]) - first)
            first += extra
            remaining -= extra
        if remaining and second < len(channels[1][1]):
            second += min(remaining, len(channels[1][1]) - second)
        budgets = [first, second]
    rendered = []
    for (label, value), budget in zip(channels, budgets, strict=True):
        visible = value if len(value) <= budget else value[-budget:] if budget else ""
        rendered.append(label + visible)
    return header + "\n" + "\n\n".join(rendered) + suffix


__all__ = ["LocalSandbox"]
