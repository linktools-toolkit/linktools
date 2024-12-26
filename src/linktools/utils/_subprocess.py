#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import errno
import os
import queue
import shlex
import subprocess
import threading
from typing import AnyStr, Optional, IO, Any, Dict, Union, List, Iterable, Generator, Tuple

import psutil

from .. import utils
from .._environ import environ
from ..decorator import cached_property, timeoutable
from ..types import TimeoutType, PathType, Timeout

STDOUT = 1
STDERR = 2


def list2cmdline(args: Iterable[str]) -> str:
    return subprocess.list2cmdline(args)


def cmdline2list(cmdline: str) -> List[str]:
    return shlex.split(cmdline)


if utils.is_unix_like():

    class Output:

        def __init__(self, stdout: IO[AnyStr], stderr: IO[AnyStr]):
            self._stdout = stdout
            self._stderr = stderr

        def get(self, timeout: Timeout):
            import select

            fds = []
            if self._stdout:
                fds.append(self._stdout)
            if self._stderr:
                fds.append(self._stderr)

            while len(fds) > 0:
                remain = utils.coalesce(timeout.remain, 1)
                if remain <= 0:  # 超时
                    break
                rlist, wlist, xlist = select.select(fds, [], [], min(remain, 1))
                if self._stdout and self._stdout in rlist:
                    try:
                        data = self._stdout.readline()
                    except OSError as e:
                        if e.errno != errno.EBADF:
                            environ.logger.debug(f"Read stdout error: {e}")
                        data = None
                    if not data:
                        fds.remove(self._stdout)
                    else:
                        yield STDOUT, data
                if self._stderr and self._stderr in rlist:
                    try:
                        data = self._stderr.readline()
                    except OSError as e:
                        if e.errno != errno.EBADF:
                            environ.logger.debug(f"Read stderr error: {e}")
                        data = None
                    if not data:
                        fds.remove(self._stderr)
                    else:
                        yield STDERR, data

else:

    class Output:

        def __init__(self, stdout: IO[AnyStr], stderr: IO[AnyStr]):
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

        def _iter_lines(self, io: IO[AnyStr], code: int, event: threading.Event):
            try:
                while True:
                    data = io.readline()
                    if not data:
                        break
                    self._queue.put((code, data))
            except OSError as e:
                if e.errno != errno.EBADF:
                    environ.logger.debug(f"Handle output error: {e}")
            finally:
                event.set()
                self._queue.put((None, None))

        def get(self, timeout: Timeout):
            while self.is_alive:
                remain = utils.coalesce(timeout.remain, 1)
                if remain <= 0:  # 超时
                    break
                try:
                    # 给个1秒超时时间防止有多个线程消费的时候死锁
                    code, data = self._queue.get(timeout=min(remain, 1))
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    pass

            while True:
                try:
                    # 需要把剩余可消费的数据消费完
                    code, data = self._queue.get_nowait()
                    if code is not None:
                        yield code, data
                except queue.Empty:
                    break


class Process(subprocess.Popen):

    @timeoutable
    def call(self, timeout: TimeoutType = None) -> int:
        with self:
            try:
                return utils.wait_process(self, timeout.remain)
            except Exception:
                self.recursive_kill()
                raise

    @timeoutable
    def check_call(self, timeout: TimeoutType = None) -> int:
        with self:
            try:
                retcode = utils.wait_process(self, timeout.remain)
                if retcode:
                    raise subprocess.CalledProcessError(retcode, self.args)
                return retcode
            except:
                self.recursive_kill()
                raise

    @timeoutable
    def fetch(self, timeout: TimeoutType = None) -> "Generator[Tuple[Optional[AnyStr], Optional[AnyStr]], Any, Any]":
        """
        获取进程的输出内容
        :param timeout: 超时时间
        :return: 返回stdout输出内容和stderr错误内容
        """
        if self.stdout or self.stderr:

            for code, data in self._output.get(timeout):
                out = err = None
                if code == STDOUT:
                    out = data
                elif code == STDERR:
                    err = data
                yield out, err
        else:

            try:
                utils.wait_process(self, timeout)
            except subprocess.TimeoutExpired:
                pass

    def recursive_kill(self) -> None:
        try:
            for p in reversed(psutil.Process(self.pid).children(recursive=True)):
                try:
                    p.terminate()
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    environ.logger.debug(f"Kill children process failed: {e}")
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            environ.logger.debug(f"List children process failed: {e}")

        self.terminate()

    @cached_property
    def _output(self):
        return Output(self.stdout, self.stderr)


def popen(
        *args: Any,
        capture_output: bool = False,
        stdin: Union[int, IO] = None, stdout: Union[int, IO] = None, stderr: Union[int, IO] = None,
        shell: bool = False, cwd: PathType = None,
        env: Dict[str, str] = None, append_env: Dict[str, str] = None, default_env: Dict[str, str] = None,
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
            cwd = environ.temp_path
            cwd.mkdir(parents=True, exist_ok=True)

    if append_env or default_env:
        env = dict(env) if env else dict(os.environ)
        if default_env:
            for key, value in default_env.items():
                env.setdefault(key, value)
        if append_env:
            env.update(append_env)

    environ.logger.debug(f"Exec cmdline: {list2cmdline(args)}")

    return Process(
        args,
        stdin=stdin, stdout=stdout, stderr=stderr,
        shell=shell, cwd=cwd,
        env=env,
        **kwargs
    )
