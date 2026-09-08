#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bubblewrap worker for the fixed SandboxSession protocol."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import errno
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import AIError, ErrorCode
from ._local_sandbox import _LocalSandboxSession
from ._sandbox import SandboxResource
from ._sandbox_protocol import (
    PROTOCOL_VERSION,
    WORKER_EXIT_CLEANUP_FAILED,
    WORKER_EXIT_OK,
    WORKER_EXIT_SESSION_FAILED,
    WORKER_BUILD,
    SandboxProtocolError,
    encode_frame,
    protocol_error,
    read_frame,
    validate_request_params,
)

_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38
_MAX_ACTIVE_REQUESTS = 256
_SAFE_ERROR_BYTES = 8 * 1024


async def main_async(arguments: argparse.Namespace) -> int:
    _validate_worker(arguments.worker_build)
    resources = _resources(arguments.resources_json)
    session = _LocalSandboxSession(
        Path("/workspace"),
        resources,
        lock_root=Path(arguments.lock_root),
    )
    reader, writer = await _stdio_streams()
    write_lock = asyncio.Lock()
    tasks: set[asyncio.Task[None]] = set()
    stop_event = asyncio.Event()
    failure_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    _reap_adopted_children(session)
    reaper_task = asyncio.create_task(
        _reap_children(session, stop_event, failure_event),
        name="sandbox-worker-reaper",
    )
    try:
        await _send_frame(
            writer,
            {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "worker_build": WORKER_BUILD,
                "status": "ready",
            },
            write_lock,
        )
        receive_task = asyncio.create_task(
            _receive_requests(
                reader,
                writer,
                write_lock,
                session,
                tasks,
                stop_event,
                failure_event,
            ),
            name="sandbox-worker-receive",
        )
        stop_task = asyncio.create_task(
            stop_event.wait(),
            name="sandbox-worker-stop",
        )
        done, _ = await asyncio.wait(
            (receive_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done and not receive_task.done():
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
        elif receive_task in done:
            receive_task.result()
        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        if failure_event.is_set():
            return WORKER_EXIT_SESSION_FAILED
    except (BrokenPipeError, ConnectionError, OSError):
        return WORKER_EXIT_SESSION_FAILED
    except SandboxProtocolError:
        return WORKER_EXIT_SESSION_FAILED
    finally:
        reaper_task.cancel()
        await asyncio.gather(reaper_task, return_exceptions=True)
        for task in tuple(tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await session.close()
        except Exception:
            return WORKER_EXIT_CLEANUP_FAILED
        writer.close()
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionError, OSError):
            pass
    return WORKER_EXIT_OK


async def _reap_children(
    session: _LocalSandboxSession,
    stop_event: asyncio.Event,
    failure_event: asyncio.Event | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            _reap_adopted_children(session)
        except Exception:
            if failure_event is not None:
                failure_event.set()
            stop_event.set()
            return
        try:
            await asyncio.wait_for(stop_event.wait(), 0.1)
        except asyncio.TimeoutError:
            continue


def _reap_adopted_children(session: _LocalSandboxSession) -> None:
    waitid = getattr(os, "waitid", None)
    if waitid is None or not hasattr(os, "P_PID"):
        return
    managed = session.managed_process_ids()
    for child_id in _child_process_ids():
        if child_id in managed:
            continue
        try:
            result = waitid(
                os.P_PID,
                child_id,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError:
            continue
        except OSError as error:
            if error.errno in {errno.ECHILD, errno.EINVAL}:
                continue
            raise
        if result is None or result.si_pid == 0:
            continue
        try:
            os.waitpid(child_id, os.WNOHANG)
        except ChildProcessError:
            continue


def _child_process_ids() -> tuple[int, ...]:
    path = Path("/proc/self/task") / str(os.getpid()) / "children"
    try:
        values = path.read_text(encoding="ascii").split()
    except OSError as error:
        raise RuntimeError("worker child list is unavailable") from error
    result: list[int] = []
    for value in values:
        try:
            child_id = int(value)
        except ValueError as error:
            raise RuntimeError("worker child list is invalid") from error
        if child_id <= 0:
            raise RuntimeError("worker child list contains an invalid pid")
        result.append(child_id)
    return tuple(result)


async def _receive_requests(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    write_lock: asyncio.Lock,
    session: _LocalSandboxSession,
    tasks: set[asyncio.Task[None]],
    stop_event: asyncio.Event,
    failure_event: asyncio.Event | None = None,
) -> None:
    active: set[str] = set()
    active_business: set[str] = set()
    active_control: set[str] = set()
    while True:
        frame = await read_frame(reader)
        if frame is None:
            return
        request_id = frame.get("request_id")
        method = frame.get("method")
        params = frame.get("params")
        if (
            isinstance(frame.get("version"), bool)
            or frame.get("version") != PROTOCOL_VERSION
            or not isinstance(request_id, str)
            or not request_id
            or len(request_id) > 128
            or not isinstance(method, str)
            or not isinstance(params, Mapping)
            or set(frame) != {"version", "request_id", "method", "params"}
            or request_id in active
        ):
            raise SandboxProtocolError("request shape is invalid")
        try:
            validate_request_params(method, params)
        except AIError as error:
            await _send_error(writer, request_id, error.code, write_lock)
            continue
        is_control = method == "stop_command"
        if (not is_control and len(active_business) >= _MAX_ACTIVE_REQUESTS) or (
            is_control and active_control
        ):
            await _send_error(
                writer,
                request_id,
                ErrorCode.TOO_MANY_PENDING_OPERATIONS,
                write_lock,
            )
            continue
        active.add(request_id)
        active_control_set = active_control if is_control else active_business
        active_control_set.add(request_id)
        task = asyncio.create_task(
            _handle_request(
                request_id,
                method,
                params,
                writer,
                write_lock,
                session,
                active,
                active_control_set,
                stop_event,
                failure_event,
            ),
            name=f"sandbox-worker-request-{request_id}",
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)


async def _handle_request(
    request_id: str,
    method: str,
    params: Mapping[str, Any],
    writer: asyncio.StreamWriter,
    write_lock: asyncio.Lock,
    session: _LocalSandboxSession,
    active: set[str],
    active_group: set[str],
    stop_event: asyncio.Event,
    failure_event: asyncio.Event | None = None,
) -> None:
    try:
        result = await _dispatch(session, method, params)
        await _send_frame(
            writer,
            {"request_id": request_id, "result": result},
            write_lock,
        )
    except AIError as error:
        try:
            await _send_error(
                writer,
                request_id,
                error.code,
                write_lock,
                error.safe_details,
            )
        except (BrokenPipeError, ConnectionError, OSError):
            if failure_event is not None:
                failure_event.set()
            stop_event.set()
        else:
            if error.code in {
                ErrorCode.SANDBOX_SESSION_LOST,
                ErrorCode.SANDBOX_CLEANUP_FAILED,
            }:
                if failure_event is not None:
                    failure_event.set()
                stop_event.set()
    except (TypeError, KeyError, ValueError):
        try:
            await _send_error(
                writer,
                request_id,
                ErrorCode.REQUEST_FIELD_INVALID,
                write_lock,
            )
        except (BrokenPipeError, ConnectionError, OSError):
            if failure_event is not None:
                failure_event.set()
            stop_event.set()
    except (BrokenPipeError, ConnectionError, OSError):
        if failure_event is not None:
            failure_event.set()
        stop_event.set()
        return
    except Exception:
        if failure_event is not None:
            failure_event.set()
        stop_event.set()
        try:
            await _send_error(
                writer,
                request_id,
                ErrorCode.SANDBOX_SESSION_LOST,
                write_lock,
            )
        except (BrokenPipeError, ConnectionError, OSError):
            pass
    finally:
        active.discard(request_id)
        active_group.discard(request_id)


async def _dispatch(
    session: _LocalSandboxSession,
    method: str,
    params: Mapping[str, Any],
) -> str:
    if method == "read_file":
        return await session.read_file(
            params["path"],
            offset=params.get("offset", 0),
            limit=params.get("limit"),
        )
    if method == "write_file":
        return await session.write_file(
            params["path"],
            params["content"],
            expected_hash=params.get("expected_hash"),
        )
    if method == "edit_file":
        return await session.edit_file(
            params["path"],
            params["old_text"],
            params["new_text"],
            expected_hash=params.get("expected_hash"),
        )
    if method == "list_directory":
        return await session.list_directory(params.get("path", "."))
    if method == "search_files":
        return await session.search_files(
            params["pattern"],
            path=params.get("path", "."),
            include_glob=params.get("include_glob"),
        )
    if method == "find_files":
        return await session.find_files(params["pattern"], path=params.get("path", "."))
    if method == "create_directory":
        return await session.create_directory(params["path"])
    if method == "file_info":
        return await session.file_info(params["path"])
    if method == "run_command":
        return await session.run_command(
            params["command"],
            timeout_seconds=params.get("timeout_seconds"),
        )
    if method == "start_command":
        return await session.start_command(params["command"])
    if method == "check_command":
        return await session.check_command(params["command_id"])
    if method == "stop_command":
        return await session.stop_command(params["command_id"])
    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


async def _send_error(
    writer: asyncio.StreamWriter,
    request_id: str,
    code: ErrorCode,
    write_lock: asyncio.Lock,
    details: Mapping[str, Any] | None = None,
) -> None:
    safe_details = dict(details or {})
    try:
        frame = encode_frame(
            {
                "request_id": request_id,
                "error": {"code": code.value, "safe_details": safe_details},
            }
        )
        if len(frame) > _SAFE_ERROR_BYTES:
            raise ValueError("error details are too large")
    except (AIError, SandboxProtocolError, TypeError, ValueError):
        frame = encode_frame(
            {
                "request_id": request_id,
                "error": protocol_error(code, reason="operation failed"),
            }
        )
    async with write_lock:
        writer.write(frame)
        await writer.drain()


async def _send_frame(
    writer: asyncio.StreamWriter,
    value: Mapping[str, Any],
    write_lock: asyncio.Lock,
) -> None:
    frame = encode_frame(value)
    async with write_lock:
        writer.write(frame)
        await writer.drain()


async def _stdio_streams() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: reader_protocol, sys.stdin.buffer)
    writer_protocol = asyncio.streams.FlowControlMixin()
    transport, _ = await loop.connect_write_pipe(
        lambda: writer_protocol,
        sys.stdout.buffer,
    )
    writer = asyncio.StreamWriter(transport, writer_protocol, reader, loop)
    return reader, writer


def _resources(value: str) -> tuple[SandboxResource, ...]:
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("worker resources are invalid") from error
    if not isinstance(raw, list):
        raise RuntimeError("worker resources are invalid")
    resources: list[SandboxResource] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise RuntimeError("worker resource is invalid")
        if set(item) != {"key", "path"}:
            raise RuntimeError("worker resource is invalid")
        key = item.get("key")
        path = item.get("path")
        if not isinstance(key, str) or key in seen or not isinstance(path, str):
            raise RuntimeError("worker resource is invalid")
        if path != f"/skills/{key}":
            raise RuntimeError("worker resource is invalid")
        resources.append(SandboxResource(key, Path(path)))
        seen.add(key)
    return tuple(resources)


def _validate_worker(build: object) -> None:
    if build != WORKER_BUILD:
        raise RuntimeError("worker build is unsupported")
    if os.getpid() != 1:
        raise RuntimeError("worker must be the namespace init process")
    _set_dumpable(False)
    if _effective_capabilities() != 0:
        raise RuntimeError("worker has capabilities")
    _set_no_new_privileges()
    if not _no_new_privileges():
        raise RuntimeError("worker has new privileges enabled")


def _set_dumpable(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_DUMPABLE, int(enabled), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "unable to set dumpability")


def _set_no_new_privileges() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "unable to set no-new-privileges")


def _effective_capabilities() -> int:
    try:
        text = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError as error:
        raise RuntimeError("worker capability status is unavailable") from error
    for line in text.splitlines():
        if line.startswith("CapEff:"):
            try:
                return int(line.split()[1], 16)
            except (IndexError, ValueError) as error:
                raise RuntimeError("worker capability status is invalid") from error
    raise RuntimeError("worker capability status is missing")


def _no_new_privileges() -> bool:
    try:
        text = Path("/proc/self/status").read_text(encoding="ascii")
    except OSError as error:
        raise RuntimeError("worker privilege status is unavailable") from error
    for line in text.splitlines():
        if line.startswith("NoNewPrivs:"):
            return line.split()[1:] == ["1"]
    raise RuntimeError("worker privilege status is missing")


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for value in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(value, stop_event.set)
        except (NotImplementedError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "worker termination signal handlers are unavailable"
            ) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources-json", required=True)
    parser.add_argument("--worker-build", required=True)
    parser.add_argument("--lock-root", required=True)
    arguments = parser.parse_args(argv)
    try:
        return asyncio.run(main_async(arguments))
    except BaseException as error:
        print(
            f"sandbox worker failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return WORKER_EXIT_SESSION_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
