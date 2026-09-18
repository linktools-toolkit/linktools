#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local sandbox process lifecycle and platform process-tree helpers."""

import asyncio
import codecs
import ctypes
import errno
import os
import signal
import sys
from collections import deque
from pathlib import Path

from ..errors import AIError, ErrorCode

_MAX_OUTPUT_CHARS = 50_000
_MAX_RENDERED_CHARS = 50_000


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
