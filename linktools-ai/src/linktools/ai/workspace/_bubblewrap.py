#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linux Bubblewrap implementation of the workspace sandbox contract."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ._sandbox import (
    SandboxResource,
    SandboxSession,
    normalize_workspace_path,
)
from ._sandbox_protocol import (
    GUARDIAN_EXIT_OK,
    GUARDIAN_EXIT_SESSION_FAILED,
    PROTOCOL_VERSION,
    WORKER_BUILD,
    SandboxProtocolError,
    encode_frame,
    read_frame,
    validate_request_params,
)

_logger = environ.get_logger("ai.workspace.bubblewrap")
_GUARDIAN_MODULE = "linktools.ai.workspace.sandbox_guardian"
_WORKER_MODULE = "linktools.ai.workspace.sandbox_worker"
_HANDSHAKE_TIMEOUT_SECONDS = 10.0
_CLOSE_TIMEOUT_SECONDS = 5.0
_MAX_PENDING_REQUESTS = 256
_MAX_RESULT_CHARS = 50_000
_MAX_SAFE_DETAILS_BYTES = 8 * 1024
_MOUNTPOINTS = (
    "workspace",
    "skills",
    "__linktools_locks",
    "proc",
    "dev",
    "tmp",
    "home/sandbox",
    "run",
    "sys",
)
_TMPFS_SIZES = (
    ("/skills", 1 * 1024 * 1024),
    ("/tmp", 64 * 1024 * 1024),
    ("/home/sandbox", 16 * 1024 * 1024),
    ("/run", 16 * 1024 * 1024),
)
_VERSION_PATTERN = re.compile(r"(?:^|\s)(\d+)\.(\d+)\.(\d+)(?:\s|$)")


class BubblewrapSandbox:
    """Open a worker in a fixed Linux namespace and mount layout."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        bwrap_executable: Path,
        hidden_paths: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(runtime_root, Path) or not runtime_root.is_absolute():
            raise ValueError("runtime_root must be an absolute Path")
        if not isinstance(bwrap_executable, Path) or not bwrap_executable.is_absolute():
            raise ValueError("bwrap_executable must be an absolute Path")
        if not isinstance(hidden_paths, tuple):
            raise TypeError("hidden_paths must be a tuple")
        self._runtime_root = runtime_root
        self._bwrap_executable = bwrap_executable
        self._hidden_paths = _normalize_hidden_paths(hidden_paths)

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        if sys.platform != "linux" or os.geteuid() == 0:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        if not hasattr(os, "pidfd_open"):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        normalized_root = _resolve_directory(root)
        runtime_root = _resolve_directory(self._runtime_root)
        bwrap = _resolve_executable(self._bwrap_executable)
        hidden_paths = _prepare_hidden_paths(normalized_root, self._hidden_paths)
        lock_root = normalized_root / ".linktools" / "locks"
        _prepare_directory(lock_root)
        _validate_rootfs(runtime_root, normalized_root)
        _validate_bwrap(bwrap)
        normalized_resources = _validate_resources(
            normalized_root,
            runtime_root,
            resources,
        )
        runtime_pidfd = _open_runtime_pidfd()
        process: asyncio.subprocess.Process | None = None
        try:
            config = _guardian_config(
                root=normalized_root,
                runtime_root=runtime_root,
                bwrap=bwrap,
                lock_root=lock_root,
                resources=normalized_resources,
                hidden_paths=hidden_paths,
            )
            process = await _spawn_guardian(config, runtime_pidfd)
            _close_fd(runtime_pidfd)
            runtime_pidfd = -1
            session = _BubblewrapSandboxSession(
                process,
                {
                    resource.key: f"/skills/{resource.key}"
                    for resource in normalized_resources
                },
            )
            try:
                await session._open_handshake()
            except BaseException:
                await session._stop_background_tasks()
                raise
            _logger.info(
                "bubblewrap sandbox session opened: root=%s resources=%s",
                normalized_root,
                tuple(resource.key for resource in normalized_resources),
            )
            return session
        except asyncio.CancelledError:
            if process is not None:
                await _abort_guardian_logged(process)
            raise
        except AIError:
            if process is not None:
                await _abort_guardian_logged(process)
            raise
        except (OSError, ValueError, RuntimeError, SandboxProtocolError) as error:
            if process is not None:
                await _abort_guardian_logged(process)
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        finally:
            if runtime_pidfd >= 0:
                _close_fd(runtime_pidfd)


class _BubblewrapSandboxSession:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        resources: Mapping[str, str],
    ) -> None:
        self._process = process
        self._resources = dict(resources)
        self._state = "OPENING"
        self._state_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[str, tuple[asyncio.Future[Any], bool]] = {}
        self._pending_lock = asyncio.Lock()
        self._lost_cleanup_task: asyncio.Task[None] | None = None
        self._lost_cleanup_error: Exception | None = None

    async def _open_handshake(self) -> None:
        stdout = self._process.stdout
        stderr = self._process.stderr
        if stdout is None or stderr is None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(stderr),
            name="bubblewrap-sandbox-stderr",
        )
        try:
            frame = await asyncio.wait_for(
                read_frame(stdout),
                _HANDSHAKE_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, SandboxProtocolError, OSError) as error:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
        if not _is_ready_frame(frame):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        self._state = "OPEN"
        self._reader_task = asyncio.create_task(
            self._read_responses(stdout),
            name="bubblewrap-sandbox-reader",
        )

    def resource_path(self, key: str) -> str:
        self._ensure_open_sync()
        if not isinstance(key, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        value = self._resources.get(key)
        if value is None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        return await self._call(
            "read_file",
            {"path": path, "offset": offset, "limit": limit},
        )

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        return await self._call(
            "write_file",
            {"path": path, "content": content, "expected_hash": expected_hash},
        )

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        return await self._call(
            "edit_file",
            {
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
                "expected_hash": expected_hash,
            },
        )

    async def list_directory(self, path: str = ".") -> str:
        return await self._call("list_directory", {"path": path})

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        return await self._call(
            "search_files",
            {"pattern": pattern, "path": path, "include_glob": include_glob},
        )

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        return await self._call("find_files", {"pattern": pattern, "path": path})

    async def create_directory(self, path: str) -> str:
        return await self._call("create_directory", {"path": path})

    async def file_info(self, path: str) -> str:
        return await self._call("file_info", {"path": path})

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        return await self._call(
            "run_command",
            {"command": command, "timeout_seconds": timeout_seconds},
        )

    async def start_command(self, command: str) -> str:
        return await self._call("start_command", {"command": command})

    async def check_command(self, command_id: str) -> str:
        return await self._call("check_command", {"command_id": command_id})

    async def stop_command(self, command_id: str) -> str:
        async with self._stop_lock:
            sent = asyncio.Event()
            task = asyncio.create_task(
                self._call(
                    "stop_command",
                    {"command_id": command_id},
                    business=False,
                    sent_event=sent,
                ),
                name="bubblewrap-sandbox-stop",
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as cancellation:
                if sent.is_set():
                    try:
                        await asyncio.shield(task)
                    except BaseException as cleanup_error:
                        if not isinstance(cleanup_error, asyncio.CancelledError):
                            _logger.exception(
                                "cancelled Bubblewrap stop cleanup failed",
                                exc_info=cleanup_error,
                            )
                        raise cancellation from cleanup_error
                else:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise cancellation

    async def close(self) -> None:
        async with self._state_lock:
            if self._state == "CLOSED":
                return
            if self._close_task is None:
                self._state = "CLOSING"
                self._close_task = asyncio.create_task(
                    self._close_impl(),
                    name="bubblewrap-sandbox-close",
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
                        "cancelled Bubblewrap sandbox cleanup failed",
                        exc_info=cleanup_error,
                    )
                raise cancellation from cleanup_error
            raise cancellation

    async def _call(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        business: bool = True,
        sent_event: asyncio.Event | None = None,
    ) -> str:
        validate_request_params(method, params)
        frame_id = uuid.uuid4().hex
        try:
            frame = encode_frame(
                {
                    "version": PROTOCOL_VERSION,
                    "request_id": frame_id,
                    "method": method,
                    "params": dict(params),
                }
            )
        except AIError:
            raise
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_future_exception)
        async with self._state_lock:
            if self._state != "OPEN":
                raise AIError(_session_state_error(self._state))
            async with self._pending_lock:
                pending_business = sum(
                    item[1] for item in self._pending.values()
                )
                if business and pending_business >= _MAX_PENDING_REQUESTS:
                    raise AIError(ErrorCode.TOO_MANY_PENDING_OPERATIONS)
                self._pending[frame_id] = (future, business)
        sent = False
        try:
            async with self._write_lock:
                if self._state != "OPEN":
                    raise AIError(_session_state_error(self._state))
                stdin = self._process.stdin
                if stdin is None or stdin.is_closing():
                    raise AIError(ErrorCode.SANDBOX_SESSION_LOST)
                stdin.write(frame)
                sent = True
                if sent_event is not None:
                    sent_event.set()
                await stdin.drain()
        except asyncio.CancelledError:
            if not sent:
                await self._drop_pending(frame_id)
            raise
        except (AIError, OSError) as error:
            if not sent:
                await self._drop_pending(frame_id)
                if (
                    isinstance(error, AIError)
                    and error.code is ErrorCode.SANDBOX_SESSION_CLOSED
                ):
                    raise
            await self._mark_lost()
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST) from error
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            raise
        if not isinstance(result, str):
            raise AIError(ErrorCode.SANDBOX_SESSION_LOST)
        return result

    async def _read_responses(self, stdout: asyncio.StreamReader) -> None:
        try:
            while True:
                frame = await read_frame(stdout)
                if frame is None:
                    break
                await self._accept_response(frame)
        except (asyncio.CancelledError, SandboxProtocolError, OSError):
            pass
        except Exception:
            _logger.exception("bubblewrap sandbox response reader failed")
        finally:
            if self._state == "OPEN":
                await self._mark_lost()

    async def _accept_response(self, frame: Mapping[str, Any]) -> None:
        request_id = frame.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise SandboxProtocolError("response request id is invalid")
        has_result = "result" in frame
        has_error = "error" in frame
        if has_result == has_error:
            raise SandboxProtocolError("response result/error is not exclusive")
        expected_keys = {"request_id", "result"} if has_result else {
            "request_id",
            "error",
        }
        if set(frame) != expected_keys:
            raise SandboxProtocolError("response fields are invalid")
        async with self._pending_lock:
            pending = self._pending.get(request_id)
        if pending is None:
            raise SandboxProtocolError("response request id is unknown")
        future, _ = pending
        if has_result:
            if (
                not isinstance(frame["result"], str)
                or len(frame["result"]) > _MAX_RESULT_CHARS
            ):
                raise SandboxProtocolError("response result is invalid")
            async with self._pending_lock:
                self._pending.pop(request_id, None)
            if not future.done():
                future.set_result(frame["result"])
            return
        error = frame["error"]
        if not isinstance(error, Mapping) or set(error) != {
            "code",
            "safe_details",
        }:
            raise SandboxProtocolError("response error is invalid")
        code_value = error.get("code")
        details = error.get("safe_details", {})
        try:
            code = ErrorCode(code_value)
            if not isinstance(details, Mapping):
                raise ValueError("safe details are invalid")
            if any(not isinstance(key, str) for key in details):
                raise ValueError("safe detail key is invalid")
            if len(
                json.dumps(
                    details,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ) > _MAX_SAFE_DETAILS_BYTES:
                raise ValueError("safe details are too large")
        except (TypeError, ValueError) as protocol_error:
            raise SandboxProtocolError("response error code is invalid") from protocol_error
        async with self._pending_lock:
            self._pending.pop(request_id, None)
        if code in {
            ErrorCode.SANDBOX_SESSION_LOST,
            ErrorCode.SANDBOX_CLEANUP_FAILED,
        }:
            if not future.done():
                future.set_exception(AIError(code, safe_details=dict(details)))
            await self._mark_lost()
        elif not future.done():
            future.set_exception(AIError(code, safe_details=dict(details)))

    async def _drop_pending(self, request_id: str) -> None:
        async with self._pending_lock:
            self._pending.pop(request_id, None)

    async def _mark_lost(self) -> None:
        async with self._state_lock:
            if self._state in {"CLOSING", "CLOSED"}:
                return
            self._state = "LOST"
            if self._lost_cleanup_task is None:
                self._lost_cleanup_task = asyncio.create_task(
                    self._terminate_lost_guardian(),
                    name="bubblewrap-sandbox-lost-cleanup",
                )
        await self._fail_pending(ErrorCode.SANDBOX_SESSION_LOST)
        _logger.error("bubblewrap sandbox session lost")

    async def _fail_pending(self, code: ErrorCode) -> None:
        async with self._pending_lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
        for future, _ in pending:
            if not future.done():
                future.set_exception(AIError(code))

    async def _terminate_lost_guardian(self) -> None:
        try:
            await _abort_guardian(
                self._process,
                allow_session_failure=True,
            )
        except Exception as error:
            self._lost_cleanup_error = error
            _logger.exception("bubblewrap lost-session cleanup failed")

    async def _close_impl(self) -> None:
        try:
            lost_cleanup = self._lost_cleanup_task
            if lost_cleanup is not None and lost_cleanup is not asyncio.current_task():
                await asyncio.shield(lost_cleanup)
                if self._lost_cleanup_error is not None:
                    raise AIError(
                        ErrorCode.SANDBOX_CLEANUP_FAILED,
                    ) from self._lost_cleanup_error
            stdin = self._process.stdin
            if stdin is not None and not stdin.is_closing():
                stdin.close()
            await _wait_guardian(
                self._process,
                allow_session_failure=True,
            )
            await self._fail_pending(ErrorCode.SANDBOX_SESSION_CLOSED)
            await self._stop_background_tasks()
            async with self._state_lock:
                self._state = "CLOSED"
            _logger.debug("bubblewrap sandbox session closed")
        except BaseException as error:
            async with self._state_lock:
                self._state = "LOST"
            await self._fail_pending(ErrorCode.SANDBOX_CLEANUP_FAILED)
            _logger.exception("bubblewrap sandbox cleanup failed")
            if isinstance(error, AIError):
                raise
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error

    async def _stop_background_tasks(self) -> None:
        current = asyncio.current_task()
        for task in (self._reader_task, self._stderr_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        while True:
            chunk = await stderr.read(4096)
            if not chunk:
                return
            _logger.warning(
                "bubblewrap guardian stderr: %s",
                chunk[:4096].decode("utf-8", "replace"),
            )

    async def _ensure_open(self) -> None:
        if self._state != "OPEN":
            raise AIError(_session_state_error(self._state))

    def _ensure_open_sync(self) -> None:
        if self._state != "OPEN":
            raise AIError(_session_state_error(self._state))


def _resolve_directory(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    if value.is_symlink() or not resolved.is_dir():
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return resolved


def _resolve_executable(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        if value.is_symlink():
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        resolved = value.resolve(strict=True)
        info = resolved.stat()
    except (OSError, RuntimeError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    return resolved


def _validate_rootfs(runtime_root: Path, workspace_root: Path) -> None:
    if (
        runtime_root == Path("/")
        or _inside(runtime_root, workspace_root)
        or _inside(workspace_root, runtime_root)
    ):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        if runtime_root.stat().st_mode & 0o222:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    except OSError as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    for relative in _MOUNTPOINTS:
        target = runtime_root / relative
        if target.is_symlink() or not target.is_dir():
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    shell = runtime_root / "bin/sh"
    python = runtime_root / "usr/bin/python3"
    if not _real_executable(shell) or not _real_executable(python):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        result = subprocess.run(
            [str(python), "--version"],
            capture_output=True,
            check=False,
            timeout=3,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    version = _parse_python_version(result.stdout + result.stderr)
    if result.returncode != 0 or version < (3, 10):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    if not _has_worker_module(runtime_root):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


def _validate_bwrap(executable: Path) -> None:
    try:
        info = executable.stat()
        if info.st_mode & (stat.S_ISUID | stat.S_ISGID):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        if _has_file_capabilities(executable):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        result = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            check=False,
            timeout=3,
            text=True,
        )
    except AIError:
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    if result.returncode != 0:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    version = _parse_bwrap_version(result.stdout + result.stderr)
    if version < (0, 12, 0):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


def _validate_resources(
    workspace_root: Path,
    runtime_root: Path,
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
        if source.is_symlink() or not resolved.is_dir():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            resolved == workspace_root
            or _inside(resolved, workspace_root)
            or resolved == workspace_root / ".linktools"
            or _inside(resolved, workspace_root / ".linktools")
            or resolved == runtime_root
            or _inside(runtime_root, resolved)
            or _inside(resolved, runtime_root)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _validate_resource_tree(resolved)
        seen.add(resource.key)
        values.append(SandboxResource(resource.key, resolved))
    return tuple(values)


def _validate_resource_tree(root: Path) -> None:
    def on_error(error: OSError) -> None:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error

    for current, directories, files in os.walk(
        root,
        followlinks=False,
        onerror=on_error,
    ):
        for name in (*directories, *files):
            target = Path(current) / name
            try:
                info = target.lstat()
            except OSError as error:
                raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
            if stat.S_ISLNK(info.st_mode):
                link = os.readlink(target)
                if Path(link).is_absolute():
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                try:
                    resolved = target.resolve(strict=True)
                    target_info = resolved.stat()
                except (OSError, RuntimeError) as error:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
                if not _inside(root, resolved) or not (
                    stat.S_ISDIR(target_info.st_mode)
                    or stat.S_ISREG(target_info.st_mode)
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            elif not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _guardian_config(
    *,
    root: Path,
    runtime_root: Path,
    bwrap: Path,
    lock_root: Path,
    resources: tuple[SandboxResource, ...],
    hidden_paths: tuple[str, ...],
) -> dict[str, Any]:
    resource_specs = [
        {"key": resource.key, "path": f"/skills/{resource.key}"}
        for resource in resources
    ]
    bwrap_args = _build_bwrap_args(
        root=root,
        runtime_root=runtime_root,
        bwrap=bwrap,
        lock_root=lock_root,
        resources=resources,
        hidden_paths=hidden_paths,
        worker_resources=resource_specs,
    )
    return {
        "version": PROTOCOL_VERSION,
        "worker_build": WORKER_BUILD,
        "bwrap_args": bwrap_args,
    }


def _build_bwrap_args(
    *,
    root: Path,
    runtime_root: Path,
    bwrap: Path,
    lock_root: Path,
    resources: tuple[SandboxResource, ...],
    hidden_paths: tuple[str, ...],
    worker_resources: list[dict[str, str]],
) -> list[str]:
    args = [
        str(bwrap),
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--disable-userns",
        "--assert-userns-disabled",
        "--new-session",
        "--die-with-parent",
        "--as-pid-1",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        str(runtime_root),
        "/",
        "--bind",
        str(root),
        "/workspace",
        "--size",
        str(_TMPFS_SIZES[0][1]),
        "--tmpfs",
        "/skills",
    ]
    for resource in resources:
        target = f"/skills/{resource.key}"
        args.extend(("--dir", target, "--ro-bind", str(resource.source), target))
        if _inside(root, resource.source):
            relative = _relative(root, resource.source)
            if not _hidden_path_covers(hidden_paths, relative):
                args.extend(
                    ("--ro-bind", str(resource.source), f"/workspace/{relative}")
                )
    for path in hidden_paths:
        target = f"/workspace/{path}"
        args.extend(("--tmpfs", target, "--remount-ro", target))
    args.extend(("--remount-ro", "/skills"))
    args.extend(("--bind", str(lock_root), "/__linktools_locks"))
    args.extend(("--proc", "/proc", "--dev", "/dev"))
    for path, size in _TMPFS_SIZES[1:]:
        args.extend(("--size", str(size), "--tmpfs", path))
    args.extend(("--tmpfs", "/sys", "--remount-ro", "/sys"))
    args.extend(
        (
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "HOME",
            "/home/sandbox",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "--setenv",
            "PYTHONIOENCODING",
            "utf-8",
            "--setenv",
            "PWD",
            "/workspace",
            "--chdir",
            "/workspace",
            "--",
            "/usr/bin/python3",
            "-I",
            "-m",
            _WORKER_MODULE,
            "--resources-json",
            json.dumps(worker_resources, ensure_ascii=False, separators=(",", ":")),
            "--worker-build",
            WORKER_BUILD,
            "--lock-root",
            "/__linktools_locks",
        )
    )
    return args


async def _spawn_guardian(
    config: Mapping[str, Any],
    runtime_pidfd: int,
) -> asyncio.subprocess.Process:
    read_fd, write_fd = os.pipe()
    process: asyncio.subprocess.Process | None = None
    try:
        payload = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(payload) > 1 * 1024 * 1024:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        os.set_inheritable(read_fd, True)
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-m",
                _GUARDIAN_MODULE,
                "--config-fd",
                str(read_fd),
                "--runtime-pidfd",
                str(runtime_pidfd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                close_fds=True,
                pass_fds=(read_fd, runtime_pidfd),
                start_new_session=True,
            ),
            name="bubblewrap-guardian-start",
        )
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError as cancellation:
            try:
                process = await asyncio.shield(spawn_task)
            except BaseException as startup_error:
                raise cancellation from startup_error
            raise cancellation
        os.set_blocking(write_fd, False)
        view = memoryview(payload)
        try:
            await _write_pipe(write_fd, view)
        finally:
            _close_fd(write_fd)
        return process
    except BaseException:
        if process is not None:
            try:
                await _abort_guardian(process)
            except BaseException as cleanup_error:
                _logger.exception(
                    "Bubblewrap guardian cleanup after startup failure failed",
                    exc_info=cleanup_error,
                )
        _close_fd(write_fd)
        raise
    finally:
        _close_fd(read_fd)


async def _abort_guardian(
    process: asyncio.subprocess.Process,
    *,
    allow_session_failure: bool = True,
) -> None:
    stdin = process.stdin
    if stdin is not None and not stdin.is_closing():
        stdin.close()
    try:
        await asyncio.wait_for(
            asyncio.shield(process.wait()),
            _CLOSE_TIMEOUT_SECONDS,
        )
        _validate_guardian_exit(
            process,
            allow_session_failure=allow_session_failure,
        )
        return
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                _CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error
        _validate_guardian_exit(
            process,
            allow_session_failure=allow_session_failure,
        )


async def _abort_guardian_logged(
    process: asyncio.subprocess.Process,
) -> None:
    try:
        await _abort_guardian(process)
    except BaseException as error:
        _logger.exception(
            "Bubblewrap guardian cleanup after open failure failed",
            exc_info=error,
        )


async def _write_pipe(fd: int, payload: memoryview) -> None:
    loop = asyncio.get_running_loop()
    offset = 0
    while offset < len(payload):
        try:
            count = os.write(fd, payload[offset:])
        except BlockingIOError:
            ready = asyncio.Event()
            loop.add_writer(fd, ready.set)
            try:
                await ready.wait()
            finally:
                loop.remove_writer(fd)
            continue
        if count <= 0:
            raise OSError("guardian configuration pipe closed")
        offset += count


async def _wait_guardian(
    process: asyncio.subprocess.Process,
    *,
    allow_session_failure: bool = False,
) -> None:
    try:
        await asyncio.wait_for(
            asyncio.shield(process.wait()),
            _CLOSE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as error:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                _CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as kill_error:
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from kill_error
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error
    _validate_guardian_exit(
        process,
        allow_session_failure=allow_session_failure,
    )


def _validate_guardian_exit(
    process: asyncio.subprocess.Process,
    *,
    allow_session_failure: bool,
) -> None:
    accepted = {GUARDIAN_EXIT_OK}
    if allow_session_failure:
        accepted.add(GUARDIAN_EXIT_SESSION_FAILED)
    if process.returncode not in accepted:
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)


def _normalize_hidden_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    normalized = {".linktools"}
    for path in paths:
        if not isinstance(path, str) or path in {"", "."}:
            raise ValueError("hidden path is invalid")
        value = normalize_workspace_path(path)
        if value == "." or value != path:
            raise ValueError("hidden path is invalid")
        normalized.add(value)
    ordered = sorted(normalized, key=lambda value: (value.count("/"), value))
    result: list[str] = []
    for value in ordered:
        if not _hidden_path_covers(tuple(result), value):
            result.append(value)
    return tuple(result)


def _prepare_hidden_paths(root: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    for path in paths:
        current = root
        for part in path.split("/"):
            current = current / part
            if current.exists():
                if current.is_symlink() or not current.is_dir():
                    raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
            else:
                _prepare_directory(current)
    return paths


def _prepare_directory(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error


def _hidden_path_covers(paths: tuple[str, ...], value: str) -> bool:
    return any(value == path or value.startswith(path + "/") for path in paths)


def _session_state_error(state: str) -> ErrorCode:
    return (
        ErrorCode.SANDBOX_SESSION_LOST
        if state == "LOST"
        else ErrorCode.SANDBOX_SESSION_CLOSED
    )


def _real_executable(path: Path) -> bool:
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and bool(info.st_mode & 0o111)


def _has_file_capabilities(path: Path) -> bool:
    if not hasattr(os, "getxattr"):
        return False
    try:
        value = os.getxattr(path, "security.capability")
    except OSError as error:
        if getattr(error, "errno", None) in {61, 93, 95}:
            return False
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error
    return bool(value)


def _has_worker_module(root: Path) -> bool:
    suffix = Path("linktools/ai/workspace/sandbox_worker.py")
    for prefix in (root / "usr", root / "opt", root / "lib"):
        if not prefix.exists():
            continue
        for current, _, files in os.walk(prefix, followlinks=False):
            if suffix.name in files and Path(current, suffix.name).as_posix().endswith(
                suffix.as_posix()
            ):
                return True
    return False


def _parse_python_version(value: str) -> tuple[int, int, int]:
    match = re.search(r"Python\s+(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return (0, 0, 0)
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def _parse_bwrap_version(value: str) -> tuple[int, int, int]:
    match = _VERSION_PATTERN.search(value)
    if match is None:
        return (0, 0, 0)
    return tuple(int(match.group(index)) for index in range(1, 4))  # type: ignore[return-value]


def _is_ready_frame(value: Mapping[str, Any] | None) -> bool:
    return bool(
        value is not None
        and set(value)
        == {"type", "protocol_version", "worker_build", "status"}
        and value.get("type") == "ready"
        and not isinstance(value.get("protocol_version"), bool)
        and value.get("protocol_version") == PROTOCOL_VERSION
        and value.get("worker_build") == WORKER_BUILD
        and value.get("status") == "ready"
    )


def _open_runtime_pidfd() -> int:
    try:
        return os.pidfd_open(os.getpid(), 0)
    except (AttributeError, OSError) as error:
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error


def _inside(parent: Path, value: Path) -> bool:
    try:
        value.relative_to(parent)
    except ValueError:
        return False
    return True


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except asyncio.CancelledError:
        pass


def _relative(root: Path, value: Path) -> str:
    return value.relative_to(root).as_posix()


def _close_fd(value: int) -> None:
    if value < 0:
        return
    try:
        os.close(value)
    except OSError:
        pass


__all__ = ["BubblewrapSandbox"]
