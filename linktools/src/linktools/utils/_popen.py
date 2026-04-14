#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import errno
import io
import os
import queue
import subprocess
import threading
from collections import ChainMap
from typing import AnyStr, Optional, IO, Any, Dict, Union, List, Iterable, Generator, Tuple

from . import _utils as utils
from ..decorator import cached_property, timeoutable
from ..types import TimeoutType, PathType, Timeout, ExecError

STDOUT = 1
STDERR = 2


def list2cmdline(args: "Iterable[str]") -> str:
    """Convert an argv list into a shell command line string.

    Args:
        args (Iterable[str]): Arguments passed to the operation.

    Returns:
        str: The operation result.
    """
    return subprocess.list2cmdline(args)


def cmdline2list(cmdline: str) -> "List[str]":
    """Split a shell command line string into an argv list.

    Args:
        cmdline (str): Command line string to write or execute.

    Returns:
        List[str]: The operation result.
    """
    import shlex
    return shlex.split(cmdline)


if utils.is_unix_like():

    class Output:

        def __init__(self, stdout: "IO[AnyStr]", stderr: "IO[AnyStr]"):
            self._stdout = stdout
            self._stderr = stderr

        def get(self, timeout: Timeout):
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
                remain = utils.coalesce(timeout.remain, 1)
                if remain <= 0:  # Timed out.
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
                        utils.get_logger().debug(f"Read io error: {e}")
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
                    utils.get_logger().debug(f"Handle output error: {e}")
            finally:
                event.set()
                self._queue.put((None, None))

        def get(self, timeout: Timeout):
            while self.is_alive:
                remain = utils.coalesce(timeout.remain, 1)
                if remain <= 0:  # Timed out.
                    break
                try:
                    # Use a one-second timeout to avoid deadlock with multiple consumers.
                    code, data = self._queue.get(timeout=min(remain, 1))
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    pass

            while True:
                try:
                    # Drain any remaining consumable data.
                    code, data = self._queue.get_nowait()
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    break


class Process(subprocess.Popen):
    """Wrapper around subprocess.Popen with timeout-aware helpers."""

    @timeoutable
    def call(self, timeout: TimeoutType = None) -> int:
        """Wait for the process and kill descendants on failure.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

        Returns:
            int: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        with self:
            try:
                return self.wait(timeout.remain)
            except:
                self.recursive_kill()
                raise

    @timeoutable
    def check_call(self, timeout: TimeoutType = None) -> int:
        """Wait for the process and raise on a nonzero exit code.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

        Returns:
            int: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        with self:
            try:
                retcode = self.wait(timeout.remain)
                if retcode:
                    raise subprocess.CalledProcessError(retcode, self.args)
                return retcode
            except:
                self.recursive_kill()
                raise

    @timeoutable
    def fetch(self, timeout: TimeoutType = None) -> "Generator[Tuple[Optional[AnyStr], Optional[AnyStr]], Any, Any]":
        """Collect stdout and stderr from the process.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

        Returns:
            Generator[Tuple[Optional[AnyStr], Optional[AnyStr]], Any, Any]: The operation result.
        """
        if self.stdout or self.stderr:
            for code, data in self._output.get(timeout):
                out = err = None
                if code == STDOUT:
                    out = data
                elif code == STDERR:
                    err = data
                yield out, err
        utils.wait_process(self, timeout)

    @timeoutable
    def exec(
            self,
            timeout: TimeoutType = None,
            ignore_errors: bool = False,
            on_stdout: "Callable[[str], None]" = None,
            on_stderr: "Callable[[str], None]" = None,
            error_type: "Callable[[str], Exception]" = ExecError
    ) -> str:
        """Run a process command until completion.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.
            ignore_errors (bool): Whether command errors should be suppressed.
            on_stdout (Callable[[str], None]): Callback invoked for stdout output.
            on_stderr (Callable[[str], None]): Callback invoked for stderr output.
            error_type (Callable[[str], Exception]): Exception type raised for command failures.

        Returns:
            str: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
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
        """Terminate the process and all child processes."""
        import psutil
        try:
            for p in reversed(psutil.Process(self.pid).children(recursive=True)):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    utils.get_logger().debug(f"Kill children process failed: {e}")
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            utils.get_logger().debug(f"List children process failed: {e}")

        self.terminate()

    @cached_property(lock=True)
    def _output(self):
        return Output(self.stdout, self.stderr)


def popen(
        *args: Any,
        capture_output: bool = False,
        stdin: "Union[int, IO]" = None, stdout: "Union[int, IO]" = None, stderr: "Union[int, IO]" = None,
        shell: bool = False, cwd: PathType = None,
        env: "Dict[str, str]" = None, append_env: "Dict[str, str]" = None, default_env: "Dict[str, str]" = None,
        **kwargs
) -> Process:
    """Create a Process for the supplied command.

    Args:
        args (Any): Arguments passed to the operation.
        capture_output (bool): The capture_output value.
        stdin (Union[int, IO]): The stdin value.
        stdout (Union[int, IO]): The stdout value.
        stderr (Union[int, IO]): The stderr value.
        shell (bool): The shell value.
        cwd (PathType): The cwd value.
        env (Dict[str, str]): The env value.
        append_env (Dict[str, str]): The append_env value.
        default_env (Dict[str, str]): The default_env value.
        kwargs: Keyword arguments passed to the operation.

    Returns:
        Process: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
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
            cwd = utils.get_environ().temp_path
            cwd.mkdir(parents=True, exist_ok=True)

    if append_env or default_env:
        maps = []
        if append_env is not None:
            maps.append(append_env)
        maps.append(env if env is not None else os.environ)
        if default_env is not None:
            maps.append(default_env)
        env = ChainMap(*maps)

    utils.get_logger().debug(f"Exec cmdline: {list2cmdline(args)}")

    return Process(
        args,
        stdin=stdin, stdout=stdout, stderr=stderr,
        shell=shell, cwd=cwd,
        env=env,
        **kwargs
    )
