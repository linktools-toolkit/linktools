#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-side Bubblewrap guardian.

The guardian deliberately treats the control stream as opaque bytes.  It owns
the Runtime pidfd and the Bubblewrap child, while the worker owns the protocol
and all workspace operations.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import select
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._sandbox_protocol import (
    GUARDIAN_EXIT_CLEANUP_FAILED,
    GUARDIAN_EXIT_OK,
    GUARDIAN_EXIT_SESSION_FAILED,
    PROTOCOL_VERSION,
    WORKER_EXIT_CLEANUP_FAILED,
    WORKER_BUILD,
)

_MAX_CONFIG_BYTES = 1 * 1024 * 1024
_MAX_BUFFER_BYTES = 16 * 1024 * 1024
_CLOSE_SECONDS = 5.0
_PR_SET_CHILD_SUBREAPER = 36
_shutdown_requested = False


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    runtime_pidfd = -1
    selector: selectors.BaseSelector | None = None
    child: subprocess.Popen[bytes] | None = None
    child_pidfd = -1
    status_fd = -1
    exit_code = GUARDIAN_EXIT_SESSION_FAILED
    try:
        runtime_pidfd = _duplicate_fd(arguments.runtime_pidfd)
        _close_fd(arguments.runtime_pidfd)
        selector = selectors.DefaultSelector()
        selector.register(runtime_pidfd, selectors.EVENT_READ, "runtime")
        if not _pidfd_is_alive(runtime_pidfd):
            raise RuntimeError("Runtime is no longer alive")
        _set_child_subreaper()
        _install_signal_handlers()
        config = _read_config(arguments.config_fd, runtime_pidfd)
        _validate_config(config)
        child, status_fd = _start_bwrap(config["bwrap_args"])
        child_pidfd = _open_child_pidfd(child.pid)
        _wait_bwrap_started(status_fd, runtime_pidfd, child)
        exit_code = _relay(
            selector,
            runtime_pidfd,
            child,
            child_pidfd,
            status_fd,
        )
    except BaseException as error:
        _write_diagnostic(error)
        exit_code = GUARDIAN_EXIT_SESSION_FAILED
    finally:
        if child is not None and child_pidfd >= 0:
            try:
                _cleanup_child(child, child_pidfd)
            except BaseException as error:
                _write_diagnostic(error)
                exit_code = GUARDIAN_EXIT_CLEANUP_FAILED
        elif child is not None:
            try:
                _cleanup_untracked_child(child)
            except BaseException as error:
                _write_diagnostic(error)
                exit_code = GUARDIAN_EXIT_CLEANUP_FAILED
        if selector is not None:
            selector.close()
        _close_fd(status_fd)
        _close_fd(runtime_pidfd)
        _close_fd(arguments.config_fd)
        _close_fd(arguments.runtime_pidfd)
    return exit_code


def _relay(
    selector: selectors.BaseSelector,
    runtime_pidfd: int,
    child: subprocess.Popen[bytes],
    child_pidfd: int,
    status_fd: int,
) -> int:
    control_in = sys.stdin.buffer
    control_out = sys.stdout.buffer
    child_in = child.stdin
    child_out = child.stdout
    child_err = child.stderr
    if child_in is None or child_out is None or child_err is None:
        raise RuntimeError("Bubblewrap pipes are unavailable")
    control_in_fd = control_in.fileno()
    control_out_fd = control_out.fileno()
    child_in_fd = child_in.fileno()
    child_out_fd = child_out.fileno()
    child_err_fd = child_err.fileno()
    for fd in (
        control_in_fd,
        control_out_fd,
        child_in_fd,
        child_out_fd,
        child_err_fd,
        status_fd,
    ):
        os.set_blocking(fd, False)
    selector.register(control_in_fd, selectors.EVENT_READ, "control-in")
    selector.register(child_out_fd, selectors.EVENT_READ, "worker-out")
    selector.register(child_err_fd, selectors.EVENT_READ, "worker-err")
    selector.register(status_fd, selectors.EVENT_READ, "bwrap-status")
    to_worker = bytearray()
    to_control = bytearray()
    control_open = True
    worker_input_open = True
    stop = False
    runtime_dead = False
    worker_failed = False
    try:
        while True:
            if _shutdown_requested:
                stop = True
            if child.poll() is not None:
                break
            if stop and worker_input_open:
                _unregister(selector, child_in_fd)
                _close_fd(child_in_fd)
                to_worker.clear()
                worker_input_open = False
            _set_write_interest(selector, child_in_fd, "worker-in", bool(to_worker))
            _set_write_interest(selector, control_out_fd, "control-out", bool(to_control))
            events = selector.select(0.25)
            if not events and child.poll() is not None:
                break
            for key, mask in events:
                tag = key.data
                if tag == "runtime":
                    stop = True
                    runtime_dead = True
                    control_open = False
                    to_control.clear()
                    _unregister(selector, control_in_fd)
                    _unregister(selector, control_out_fd)
                    _close_fd(control_in_fd)
                    _close_fd(control_out_fd)
                    _unregister(selector, child_in_fd)
                    _close_fd(child_in_fd)
                    to_worker.clear()
                    worker_input_open = False
                    continue
                if tag == "control-in" and mask & selectors.EVENT_READ:
                    data = _read_nonblocking(control_in_fd)
                    if data is None:
                        continue
                    if not data:
                        control_open = False
                        stop = True
                        to_control.clear()
                        to_worker.clear()
                        _unregister(selector, control_in_fd)
                        _close_fd(control_in_fd)
                        if worker_input_open:
                            _unregister(selector, child_in_fd)
                            _close_fd(child_in_fd)
                            worker_input_open = False
                    else:
                        to_worker.extend(data)
                        if len(to_worker) > _MAX_BUFFER_BYTES:
                            raise RuntimeError("guardian input backpressure limit exceeded")
                elif tag == "worker-out" and mask & selectors.EVENT_READ:
                    data = _read_nonblocking(child_out_fd)
                    if data is None:
                        continue
                    if not data:
                        worker_failed = control_open and not runtime_dead
                        stop = True
                        control_open = False
                        to_control.clear()
                        _unregister(selector, control_in_fd)
                        _unregister(selector, control_out_fd)
                        _close_fd(control_in_fd)
                        _close_fd(control_out_fd)
                    else:
                        if control_open and not runtime_dead:
                            to_control.extend(data)
                            if len(to_control) > _MAX_BUFFER_BYTES:
                                raise RuntimeError(
                                    "guardian output backpressure limit exceeded"
                                )
                elif tag == "bwrap-status" and mask & selectors.EVENT_READ:
                    data = _read_nonblocking(status_fd)
                    if data is None:
                        continue
                    if not data:
                        _unregister(selector, status_fd)
                        _close_fd(status_fd)
                    elif len(data) > _MAX_CONFIG_BYTES:
                        raise RuntimeError("Bubblewrap status is too large")
                elif tag == "worker-err" and mask & selectors.EVENT_READ:
                    data = _read_nonblocking(child_err_fd)
                    if data is None:
                        continue
                    if data:
                        _write_stderr(data)
                    else:
                        _unregister(selector, child_err_fd)
                        _close_fd(child_err_fd)
                elif tag == "worker-in" and mask & selectors.EVENT_WRITE:
                    if not _write_nonblocking(child_in_fd, to_worker):
                        stop = True
                        worker_input_open = False
                        to_worker.clear()
                        _unregister(selector, child_in_fd)
                        _close_fd(child_in_fd)
                        control_open = False
                        to_control.clear()
                        _unregister(selector, control_in_fd)
                        _unregister(selector, control_out_fd)
                        _close_fd(control_in_fd)
                        _close_fd(control_out_fd)
                elif tag == "control-out" and mask & selectors.EVENT_WRITE:
                    if not _write_nonblocking(control_out_fd, to_control):
                        stop = True
                        control_open = False
                        to_control.clear()
                        _unregister(selector, control_in_fd)
                        _unregister(selector, control_out_fd)
                        _close_fd(control_in_fd)
                        _close_fd(control_out_fd)
            if (runtime_dead or not control_open) and not to_control:
                break
        worker_returncode = child.poll()
        if worker_returncode == WORKER_EXIT_CLEANUP_FAILED:
            return GUARDIAN_EXIT_CLEANUP_FAILED
        if worker_returncode is not None and control_open and not runtime_dead:
            worker_failed = True
        return (
            GUARDIAN_EXIT_SESSION_FAILED
            if worker_failed
            else GUARDIAN_EXIT_OK
        )
    finally:
        for fd in (
            control_in_fd,
            control_out_fd,
            child_in_fd,
            child_out_fd,
            child_err_fd,
            status_fd,
        ):
            try:
                selector.unregister(fd)
            except (KeyError, ValueError):
                pass


def _start_bwrap(arguments: Any) -> tuple[subprocess.Popen[bytes], int]:
    if (
        not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(value, str) or not value for value in arguments)
    ):
        raise RuntimeError("Bubblewrap command is invalid")
    status_read = -1
    status_write = -1
    try:
        status_read, status_write = os.pipe()
        os.set_inheritable(status_write, True)
        try:
            separator = arguments.index("--")
        except ValueError as error:
            raise RuntimeError("Bubblewrap command has no command separator") from error
        command = [
            *arguments[:separator],
            "--json-status-fd",
            str(status_write),
            *arguments[separator:],
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(status_write,),
            start_new_session=True,
        )
        return process, status_read
    except OSError as error:
        _close_fd(status_read)
        raise RuntimeError("Bubblewrap process could not start") from error
    except BaseException:
        _close_fd(status_read)
        raise
    finally:
        _close_fd(status_write)


def _wait_bwrap_started(
    status_fd: int,
    runtime_pidfd: int,
    child: subprocess.Popen[bytes],
) -> None:
    os.set_blocking(status_fd, False)
    buffer = bytearray()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("Bubblewrap exited before startup")
        readable, _, _ = select.select(
            [status_fd, runtime_pidfd],
            [],
            [],
            min(0.25, max(0.0, deadline - time.monotonic())),
        )
        if runtime_pidfd in readable:
            raise RuntimeError("Runtime exited during Bubblewrap startup")
        if status_fd not in readable:
            continue
        data = _read_nonblocking(status_fd)
        if data is None:
            continue
        if not data:
            raise RuntimeError("Bubblewrap status closed before startup")
        buffer.extend(data)
        if len(buffer) > _MAX_CONFIG_BYTES:
            raise RuntimeError("Bubblewrap status is too large")
        while b"\n" in buffer:
            line, _, remainder = buffer.partition(b"\n")
            buffer[:] = remainder
            if _status_has_child(line):
                return
            if _status_has_exit(line):
                raise RuntimeError("Bubblewrap exited before startup")
    raise RuntimeError("Bubblewrap startup status timed out")


def _status_has_child(value: bytes) -> bool:
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Bubblewrap startup status is invalid") from error
    return (
        isinstance(decoded, dict)
        and isinstance(decoded.get("child-pid"), int)
        and not isinstance(decoded.get("child-pid"), bool)
        and decoded["child-pid"] > 0
    )


def _status_has_exit(value: bytes) -> bool:
    try:
        decoded = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Bubblewrap exit status is invalid") from error
    return isinstance(decoded, dict) and "exit-code" in decoded


def _cleanup_child(child: subprocess.Popen[bytes], child_pidfd: int) -> None:
    try:
        if child.poll() is None:
            _send_child_signal(child_pidfd, signal.SIGTERM)
            _wait_child(child, _CLOSE_SECONDS)
        if child.poll() is None:
            _send_child_signal(child_pidfd, signal.SIGKILL)
            _wait_child(child, _CLOSE_SECONDS)
        if child.poll() is None:
            raise RuntimeError("Bubblewrap did not exit")
        child.wait()
        _reap_adopted_children()
    finally:
        _close_fd(child_pidfd)


def _cleanup_untracked_child(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is None:
        try:
            child.terminate()
        except ProcessLookupError:
            pass
        _wait_child(child, _CLOSE_SECONDS)
    if child.poll() is None:
        try:
            child.kill()
        except ProcessLookupError:
            pass
        _wait_child(child, _CLOSE_SECONDS)
    if child.poll() is None:
        raise RuntimeError("Bubblewrap child did not exit during cleanup")
    child.wait()
    _reap_adopted_children()


def _wait_child(child: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)


def _reap_adopted_children() -> None:
    waitid = getattr(os, "waitid", None)
    if waitid is None or not hasattr(os, "P_PID"):
        return
    while True:
        child_ids = _child_process_ids()
        if not child_ids:
            return
        for child_id in child_ids:
            _terminate_and_reap_child(child_id, waitid)


def _terminate_and_reap_child(child_id: int, waitid: Any) -> None:
    try:
        pidfd = _open_child_pidfd(child_id)
    except OSError as error:
        if error.errno == errno.ESRCH:
            _wait_child_pid(child_id)
            return
        raise
    try:
        if not _child_has_exited(child_id, waitid):
            _send_child_pidfd_signal(pidfd, signal.SIGTERM)
            if not _wait_child_until(child_id, waitid, _CLOSE_SECONDS):
                _send_child_pidfd_signal(pidfd, signal.SIGKILL)
                if not _wait_child_until(child_id, waitid, _CLOSE_SECONDS):
                    raise RuntimeError(
                        f"adopted child {child_id} did not exit"
                    )
        _wait_child_pid(child_id)
    finally:
        _close_fd(pidfd)


def _child_has_exited(child_id: int, waitid: Any) -> bool:
    try:
        result = waitid(
            os.P_PID,
            child_id,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:
        return True
    except OSError as error:
        if error.errno in {errno.ECHILD, errno.ESRCH}:
            return True
        raise
    return result is not None and result.si_pid == child_id


def _wait_child_until(child_id: int, waitid: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _child_has_exited(child_id, waitid):
            return True
        time.sleep(0.02)
    return _child_has_exited(child_id, waitid)


def _wait_child_pid(child_id: int) -> None:
    try:
        os.waitpid(child_id, 0)
    except ChildProcessError:
        pass
    except OSError as error:
        if error.errno not in {errno.ECHILD, errno.ESRCH}:
            raise


def _child_process_ids() -> tuple[int, ...]:
    children_path = (
        "/proc"
        f"/{os.getpid()}"
        "/task"
        f"/{os.getpid()}"
        "/children"
    )
    try:
        with Path(children_path).open(encoding="ascii") as stream:
            value = stream.read()
    except OSError as error:
        raise RuntimeError("guardian child list is unavailable") from error
    child_ids: list[int] = []
    for item in value.split():
        try:
            child_id = int(item)
        except ValueError as error:
            raise RuntimeError("guardian child list is invalid") from error
        if child_id <= 0:
            raise RuntimeError("guardian child list contains an invalid pid")
        child_ids.append(child_id)
    return tuple(child_ids)


def _send_child_pidfd_signal(pidfd: int, value: signal.Signals) -> None:
    try:
        _send_pidfd_signal(pidfd, value)
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise


def _set_child_subreaper() -> None:
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
    if prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "unable to become child subreaper")


def _open_child_pidfd(pid: int) -> int:
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        raise RuntimeError("pidfd_open is unavailable")
    return pidfd_open(pid, 0)


def _send_pidfd_signal(pidfd: int, value: signal.Signals) -> None:
    sender = getattr(signal, "pidfd_send_signal", None)
    if sender is not None:
        sender(pidfd, value, None, 0)
        return
    if not sys.platform.startswith("linux"):
        raise OSError("pidfd signaling is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    syscall.restype = ctypes.c_long
    result = syscall(
        _pidfd_send_signal_number(),
        pidfd,
        int(value),
        None,
        0,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _send_child_signal(child_pidfd: int, value: signal.Signals) -> None:
    try:
        _send_pidfd_signal(child_pidfd, value)
    except OSError as error:
        if error.errno != errno.ESRCH:
            raise


def _pidfd_send_signal_number() -> int:
    machine = os.uname().machine.lower()
    if machine in {
        "aarch64",
        "arm64",
        "ppc64",
        "ppc64le",
        "riscv64",
        "s390x",
        "x86_64",
        "amd64",
    }:
        return 424
    raise OSError("pidfd signaling syscall number is unknown")


def _pidfd_is_alive(pidfd: int) -> bool:
    import select

    readable, _, _ = select.select([pidfd], [], [], 0)
    return not readable


def _read_config(fd_value: int, runtime_pidfd: int) -> dict[str, Any]:
    if fd_value < 0:
        raise RuntimeError("config fd is invalid")
    if runtime_pidfd < 0:
        raise RuntimeError("Runtime pidfd is invalid")
    os.set_blocking(fd_value, False)
    chunks: list[bytes] = []
    size = 0
    while True:
        readable, _, _ = select.select(
            [fd_value, runtime_pidfd],
            [],
            [],
            0.25,
        )
        if runtime_pidfd in readable:
            raise RuntimeError("Runtime exited during guardian startup")
        if fd_value not in readable:
            continue
        try:
            chunk = os.read(fd_value, 64 * 1024)
        except BlockingIOError:
            continue
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_CONFIG_BYTES:
            raise RuntimeError("guardian config is too large")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("guardian config is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError("guardian config is not an object")
    return value


def _validate_config(value: Mapping[str, Any]) -> None:
    arguments = value.get("bwrap_args")
    if (
        set(value) != {"version", "worker_build", "bwrap_args"}
        or isinstance(value.get("version"), bool)
        or value.get("version") != PROTOCOL_VERSION
        or value.get("worker_build") != WORKER_BUILD
        or not isinstance(arguments, list)
        or not arguments
        or any(not isinstance(argument, str) or not argument for argument in arguments)
    ):
        raise RuntimeError("guardian config version is unsupported")


def _duplicate_fd(value: int) -> int:
    if value < 0:
        raise RuntimeError("Runtime pidfd is invalid")
    try:
        return os.dup(value)
    except OSError as error:
        raise RuntimeError("Runtime pidfd cannot be duplicated") from error


def _read_nonblocking(fd: int) -> bytes | None:
    try:
        return os.read(fd, 64 * 1024)
    except BlockingIOError:
        return None
    except OSError as error:
        if error.errno in {errno.EBADF, errno.EPIPE, errno.ECONNRESET}:
            return b""
        raise


def _write_nonblocking(fd: int, buffer: bytearray) -> bool:
    if not buffer:
        return True
    try:
        count = os.write(fd, buffer)
    except BlockingIOError:
        return True
    except OSError as error:
        if error.errno in {errno.EBADF, errno.EPIPE, errno.ECONNRESET}:
            return False
        raise
    del buffer[:count]
    return True


def _set_write_interest(
    selector: selectors.BaseSelector,
    fd: int,
    tag: str,
    enabled: bool,
) -> None:
    try:
        key = selector.get_key(fd)
    except KeyError:
        if enabled:
            selector.register(fd, selectors.EVENT_WRITE, tag)
        return
    events = key.events
    wanted = events | selectors.EVENT_WRITE if enabled else events & ~selectors.EVENT_WRITE
    if wanted:
        selector.modify(fd, wanted, tag)
    else:
        selector.unregister(fd)


def _unregister(selector: selectors.BaseSelector, fd: int) -> None:
    try:
        selector.unregister(fd)
    except (KeyError, ValueError):
        pass


def _install_signal_handlers() -> None:
    global _shutdown_requested
    _shutdown_requested = False

    def request_shutdown(signum: int, frame: object) -> None:
        del signum, frame
        global _shutdown_requested
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)


def _write_stderr(value: bytes) -> None:
    try:
        stderr_fd = sys.stderr.buffer.fileno()
        os.set_blocking(stderr_fd, False)
        os.write(stderr_fd, value[:64 * 1024])
    except (BlockingIOError, OSError):
        pass


def _write_diagnostic(error: BaseException) -> None:
    _write_stderr(f"sandbox guardian failed: {type(error).__name__}\n".encode())


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-fd", type=int, required=True)
    parser.add_argument("--runtime-pidfd", type=int, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
