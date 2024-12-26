#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import functools
import threading
import time
import traceback
from collections import deque
from typing import Callable

from . import utils
from ._environ import environ
from .decorator import timeoutable
from .types import TimeoutType

_logger = environ.get_logger("reactor")


# Code stolen from frida_tools.application.Reactor
class Reactor:

    def __init__(self, on_stop=None, on_error=None):
        self._running = False
        self._on_stop = on_stop
        self._on_error = on_error
        self._pending = deque([])
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker = None

    def is_running(self) -> "bool":
        with self._lock:
            return self._running

    def start(self):
        if self._running:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(target=self._run)
            self._worker.daemon = True
            self._worker.start()

    @timeoutable
    def run(self, timeout: TimeoutType):
        with self:
            self.wait(timeout)

    def _run(self):
        running = True
        while running:
            now = time.time()
            work = None
            timeout = None
            with self._lock:
                for item in self._pending:
                    (f, when) = item
                    if now >= when:
                        work = f
                        self._pending.remove(item)
                        break
                if len(self._pending) > 0:
                    timeout = max([min(map(lambda item: item[1], self._pending)) - now, 0])
                previous_pending_length = len(self._pending)

            if work is not None:
                try:
                    work()
                except (KeyboardInterrupt, EOFError) as e:
                    if self._on_error is not None:
                        self._on_error(e, traceback.format_exc())
                    self.signal_stop()
                except BaseException as e:
                    if self._on_error is not None:
                        self._on_error(e, traceback.format_exc())
                    else:
                        _logger.warning("Reactor caught an exception", exc_info=True)

            with self._lock:
                if self._running and len(self._pending) == previous_pending_length:
                    self._cond.wait(timeout)
                running = self._running

        if self._on_stop is not None:
            self._on_stop()

    def stop(self):
        self.signal_stop()
        self.wait()

    def _stop(self):
        with self._lock:
            self._running = False

    def signal_stop(self, delay: float = None):
        self.schedule(self._stop, delay)

    def schedule(self, fn: Callable[[], any], delay: float = None):
        now = time.time()
        if delay is not None:
            when = now + delay
        else:
            when = now
        with self._lock:
            self._pending.append((functools.partial(self._work, fn), when))
            self._cond.notify()

    def _work(self, fn: Callable[[], any]):
        fn()

    @timeoutable
    def wait(self, timeout: TimeoutType = None) -> bool:
        worker = self._worker
        if worker:
            if threading.current_thread().ident == worker.ident:
                raise RuntimeError("Cannot wait on the reactor from its own thread")
            return utils.wait_thread(worker, timeout)
        return True

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
