#!/usr/bin/env python3
# -*- coding:utf-8 -*-
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


class _ReactorEvent:

    def __init__(self, fn: Callable[[], any], when: float, interval: float):
        self.fn = fn
        self.when = when
        self.interval = interval

    def copy(self, **kwargs):
        return _ReactorEvent(
            kwargs.get("fn", self.fn),
            kwargs.get("when", self.when),
            kwargs.get("interval", self.interval),
        )


# Code stolen from frida_tools.application.Reactor
class Reactor:

    def __init__(self, on_stop=None, on_error=None):
        self._running = False
        self._on_stop = on_stop
        self._on_error = on_error
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker = None
        self._pending: "deque[_ReactorEvent]" = deque([])

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
            fn = None
            timeout = None
            with self._lock:
                for item in self._pending:
                    if now >= item.when:
                        self._pending.remove(item)
                        if item.interval is not None:
                            self._pending.append(item.copy(when=item.when + item.interval))
                        fn = item.fn
                        break
                if len(self._pending) > 0:
                    timeout = max([min(map(lambda o: o.when, self._pending)) - now, 0])
                previous_pending_length = len(self._pending)

            if fn is not None:
                try:
                    self._work(fn)
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

    def schedule(self, fn: Callable[[], any], delay: float = None, interval: float = None):
        now = time.time()
        if delay is not None:
            when = now + delay
        else:
            when = now
        with self._lock:
            item = _ReactorEvent(fn, when, interval)
            self._pending.append(item)
            self._cond.notify()

    def _work(self, fn: Callable[[], any]):
        fn()

    @timeoutable
    def wait(self, timeout: TimeoutType = None) -> bool:
        worker = self._worker
        if worker:
            if threading.current_thread().ident == worker.ident:
                _logger.warning("Cannot wait on the reactor from its own thread")
                return False
            return utils.wait_thread(worker, timeout)
        return True

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
