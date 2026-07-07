#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import errno
import io
import os
import queue
import subprocess
import threading
import time as _time
from collections import ChainMap
from typing import TYPE_CHECKING

from ..decorator import cached_property, timeoutable
from ..errors import ExecError
from ..system import is_unix_like, wait_process
from ..utils import list2cmdline

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, AnyStr, Callable, IO
    from ..types import PathType, Timeout, TimeoutType

STDOUT = 1
STDERR = 2


def _coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _get_environ():
    from ..core import environ
    return environ


_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = _get_environ().get_logger("process")
    return _logger


if is_unix_like():

    class Output:

        def __init__(self, stdout: "IO[AnyStr]", stderr: "IO[AnyStr]"):
            self._stdout = stdout
            self._stderr = stderr

        def get(self, timeout: "Timeout"):
            import select

            fds = []
            stdout, stderr = None, None
            if self._stdout:
                stdout = self.IOWrapper(self._stdout, STDOUT)
                fds.append(stdout.fd)
            if self._stderr:
                stderr = self.IOWrapper(self._stderr, STDERR)
                fds.append(stderr.fd)

            while len(fds) > 0:
                remain = _coalesce(timeout.remaining, 1)
                if remain <= 0:
                    break
                rlist, wlist, xlist = select.select(fds, [], [], min(remain, 1))
                if stdout.fd is not None and stdout.fd in rlist:
                    yield from stdout.read_lines()
                    if stdout.closed:
                        fds.remove(stdout.fd)
                if stderr.fd is not None and stderr.fd in rlist:
                    yield from stderr.read_lines()
                    if stderr.closed:
                        fds.remove(stderr.fd)

            yield from stdout.read_remain_line()
            yield from stderr.read_remain_line()

        class IOWrapper:

            def __init__(self, io: "IO[AnyStr]", code: int):
                self.io = io
                self.fd = io.fileno()
                self.code = code
                self.closed = False
                self.buffer = bytearray()

            def read_lines(self):
                data = None
                try:
                    if not self.closed:
                        data = os.read(self.fd, 32768)
                        if data:
                            self.buffer.extend(data)
                except OSError as e:
                    if e.errno != errno.EBADF:
                        _get_logger().debug(f"Read io error: {e}")
                if data:
                    while True:
                        index = self.buffer.find(b"\n")
                        if index < 0:
                            break
                        self.buffer, line = self.buffer[index + 1:], self.buffer[:index + 1]
                        line = line.decode(self.io.encoding, self.io.errors) \
                            if isinstance(self.io, io.TextIOWrapper) \
                            else bytes(line)
                        yield self.code, line
                else:
                    yield from self.read_remain_line()
                    self.closed = True

            def read_remain_line(self):
                if self.buffer:
                    self.buffer, line = bytearray(), self.buffer
                    line = line.decode(self.io.encoding, self.io.errors) \
                        if isinstance(self.io, io.TextIOWrapper) \
                        else bytes(line)
                    yield self.code, line


else:

    class Output:

        def __init__(self, stdout: "IO[AnyStr]", stderr: "IO[AnyStr]"):
            self._queue = queue.Queue()
            self._stdout_finished = None
            self._stdout_thread = None
            self._stderr_finished = None
            self._stderr_thread = None
            if stdout:
                self._stdout_finished = threading.Event()
                self._stdout_thread = threading.Thread(
                    target=self._iter_lines,
                    args=(stdout, STDOUT, self._stdout_finished,)
                )
                self._stdout_thread.daemon = True
                self._stdout_thread.start()
            if stderr:
                self._stderr_finished = threading.Event()
                self._stderr_thread = threading.Thread(
                    target=self._iter_lines,
                    args=(stderr, STDERR, self._stderr_finished,)
                )
                self._stderr_thread.daemon = True
                self._stderr_thread.start()

        @property
        def is_alive(self):
            if self._stdout_finished and not self._stdout_finished.is_set():
                return True
            if self._stderr_finished and not self._stderr_finished.is_set():
                return True
            return False

        def _iter_lines(self, io: "IO[AnyStr]", code: int, event: "threading.Event"):
            try:
                while True:
                    data = io.readline()
                    if not data:
                        break
                    self._queue.put((code, data))
            except OSError as e:
                if e.errno != errno.EBADF:
                    _get_logger().debug(f"Handle output error: {e}")
            finally:
                event.set()
                self._queue.put((None, None))

        def get(self, timeout: "Timeout"):
            while self.is_alive:
                remain = _coalesce(timeout.remaining, 1)
                if remain <= 0:
                    break
                try:
                    code, data = self._queue.get(timeout=min(remain, 1))
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    pass

            while True:
                try:
                    code, data = self._queue.get_nowait()
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    break


class ProcessResult(object):
    """Result of a completed process (v2 §12.1 RUN-PROC-002)."""

    def __init__(self, args, returncode, stdout=None, stderr=None, duration=None, timed_out=False):
        # type: (Any, int, Any, Any, float, bool) -> None
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.timed_out = timed_out

    @property
    def succeeded(self):
        return self.returncode == 0

    def __repr__(self):
        return "ProcessResult(args=%r, returncode=%d, duration=%.2fs)" % (
            self.args, self.returncode, self.duration or 0)


class Process(object):
    """Composition wrapper around subprocess.Popen (v2 §12.1 RUN-PROC-001).

    No longer inherits Popen; instead wraps it and delegates all attribute
    access via __getattr__. This means callers that used Popen methods/attrs
    (wait, poll, kill, stdout, pid, returncode, ...) work unchanged through
    delegation, while Process owns its own lifecycle methods (call, check_call,
    fetch, exec, recursive_kill, wait_for_result).
    """

    def __init__(self, *args, **kwargs):
        self._popen = subprocess.Popen(*args, **kwargs)
        self._started_at = _time.monotonic()

    def __getattr__(self, name):
        # Delegate any attribute not defined on Process to the wrapped Popen.
        # This covers wait, poll, kill, terminate, communicate, stdout, stderr,
        # stdin, pid, returncode, args, etc.
        return getattr(self._popen, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._popen.wait()

    @classmethod
    def start(cls, *args, **kwargs):
        # type: (**Any) -> Process
        """Create and start a Process (v2 §12.1)."""
        return cls(*args, **kwargs)

    def wait_for_result(self, timeout=None):
        # type: (TimeoutType) -> ProcessResult
        """Wait for the process and return a ProcessResult (v2 §12.1)."""
        from ..types import Timeout
        timed_out = False
        wall = None
        if timeout is not None:
            t = Timeout(timeout)
            wall = t.remaining
        try:
            retcode = self._popen.wait(timeout=wall)
        except subprocess.TimeoutExpired:
            self.recursive_kill()
            retcode = self._popen.returncode
            timed_out = True
        duration = _time.monotonic() - self._started_at
        return ProcessResult(
            args=self._popen.args,
            returncode=retcode,
            stdout=self._popen.stdout,
            stderr=self._popen.stderr,
            duration=duration,
            timed_out=timed_out,
        )

    @timeoutable
    def call(self, timeout: "TimeoutType" = None) -> int:
        with self:
            try:
                return self._popen.wait(timeout.remaining)
            except Exception:
                self.recursive_kill()
                raise

    @timeoutable
    def check_call(self, timeout: "TimeoutType" = None) -> int:
        with self:
            try:
                retcode = self._popen.wait(timeout.remaining)
                if retcode:
                    raise subprocess.CalledProcessError(retcode, self._popen.args)
                return retcode
            except Exception:
                self.recursive_kill()
                raise

    @timeoutable
    def fetch(self, timeout: "TimeoutType" = None) -> "Generator[tuple[AnyStr | None, AnyStr | None], Any, Any]":
        if self.stdout or self.stderr:
            for code, data in self._output.get(timeout):
                out = err = None
                if code == STDOUT:
                    out = data
                elif code == STDERR:
                    err = data
                yield out, err
        wait_process(self, timeout)

    @timeoutable
    def exec(
            self,
            timeout: "TimeoutType" = None,
            ignore_errors: bool = False,
            on_stdout: "Callable[[str], None]" = None,
            on_stderr: "Callable[[str], None]" = None,
            error_type: "Callable[[str], Exception]" = ExecError
    ) -> str:
        try:
            out = err = None
            for _out, _err in self.fetch(timeout=timeout):
                if _out is not None:
                    out = _out if out is None else out + _out
                    if on_stdout:
                        data: str = _out.decode(errors="ignore") if isinstance(_out, bytes) else _out
                        data = data.rstrip()
                        if data:
                            on_stdout(data)
                if _err is not None:
                    err = _err if err is None else err + _err
                    if on_stderr:
                        data: str = _err.decode(errors="ignore") if isinstance(_err, bytes) else _err
                        data = data.rstrip()
                        if data:
                            on_stderr(data)

            if not ignore_errors and self.poll() not in (0, None):
                if isinstance(err, bytes):
                    err = err.decode(errors="ignore")
                    err = err.strip()
                elif isinstance(err, str):
                    err = err.strip()
                if err:
                    raise error_type(err)

            if isinstance(out, bytes):
                out = out.decode(errors="ignore")
                out = out.strip()
            elif isinstance(out, str):
                out = out.strip()

            return out or ""

        finally:
            self.recursive_kill()

    def recursive_kill(self) -> None:
        import psutil
        try:
            for p in reversed(psutil.Process(self.pid).children(recursive=True)):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    _get_logger().debug(f"Kill children process failed: {e}")
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            _get_logger().debug(f"List children process failed: {e}")

        self.terminate()

    @cached_property(lock=True)
    def _output(self):
        return Output(self.stdout, self.stderr)


def popen(
        *args: "Any",
        capture_output: bool = False,
        stdin: "int | IO" = None, stdout: "int | IO" = None, stderr: "int | IO" = None,
        shell: bool = False, cwd: "PathType" = None,
        env: "dict[str, str]" = None, append_env: "dict[str, str]" = None, default_env: "dict[str, str]" = None,
        **kwargs
) -> Process:
    args = [str(arg) for arg in args]

    if capture_output is True:
        if stdout is not None or stderr is not None:
            raise ValueError("stdout and stderr arguments may not be used "
                             "with capture_output.")
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE

    if not cwd:
        try:
            cwd = os.getcwd()
        except FileNotFoundError:
            cwd = _get_environ().temp_path
            cwd.mkdir(parents=True, exist_ok=True)

    if append_env or default_env:
        maps = []
        if append_env is not None:
            maps.append(append_env)
        maps.append(env if env is not None else os.environ)
        if default_env is not None:
            maps.append(default_env)
        env = ChainMap(*maps)

    _get_logger().debug(f"Exec cmdline: {list2cmdline(args)}")

    return Process(
        args,
        stdin=stdin, stdout=stdout, stderr=stderr,
        shell=shell, cwd=cwd,
        env=env,
        **kwargs
    )
