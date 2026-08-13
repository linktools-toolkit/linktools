#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections as _collections
import threading as _threading
import time as _time
import typing as _t

from ..system import wait_thread
from ..types import Timeout

if _t.TYPE_CHECKING:
    from ..types import TimeoutType


_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        from ..core import environ
        _logger = environ.get_logger("reactor")
    return _logger


class _ReactorEvent:

    def __init__(self, fn: "_t.Callable[[], _t.Any]", when: float, interval: float):
        self.fn = fn
        self.when = when
        self.interval = interval

    def copy(self, **kwargs: "_t.Any") -> "_ReactorEvent":
        return _ReactorEvent(
            kwargs.get("fn", self.fn),
            kwargs.get("when", self.when),
            kwargs.get("interval", self.interval),
        )


class Reactor:

    def __init__(self, on_stop: "_t.Callable | None" = None, on_error: "_t.Callable[..., _t.Any] | None" = None) -> None:
        self._running = False
        self._on_stop = on_stop
        self._on_error = on_error
        self._lock = _threading.Lock()
        self._cond = _threading.Condition(self._lock)
        self._worker = None
        self._pending: "_collections.deque[_ReactorEvent]" = _collections.deque([])

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        if self._running:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = _threading.Thread(target=self._run)
            self._worker.daemon = True
            self._worker.start()

    def run(self, timeout: "TimeoutType") -> None:
        with self:
            self.wait(Timeout(timeout))

    def _run(self):
        running = True
        while running:
            now = _time.monotonic()  # never wall-clock
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
                        import traceback
                        self._on_error(e, traceback.format_exc())
                    self.signal_stop()
                except Exception as e:  # never swallow KI/SystemExit/GeneratorExit
                    if self._on_error is not None:
                        import traceback
                        self._on_error(e, traceback.format_exc())
                    else:
                        _get_logger().warning("Reactor caught an exception", exc_info=True)

            with self._lock:
                if self._running and len(self._pending) == previous_pending_length:
                    self._cond.wait(timeout)
                running = self._running

        if self._on_stop is not None:
            self._on_stop()

    def stop(self) -> None:
        self.signal_stop()
        self.wait()

    def _stop(self) -> None:
        with self._lock:
            self._running = False

    def signal_stop(self, delay: "float | None" = None) -> None:
        self.schedule(self._stop, delay)

    def schedule(self, fn: "_t.Callable[[], _t.Any]", delay: "float | None" = None, interval: "float | None" = None) -> None:
        now = _time.monotonic()
        when = now + delay if delay is not None else now
        with self._lock:
            item = _ReactorEvent(fn, when, interval)
            self._pending.append(item)
            self._cond.notify()

    def _work(self, fn: "_t.Callable[[], _t.Any]"):
        fn()

    def wait(self, timeout: "TimeoutType | None" = None) -> bool:
        worker = self._worker
        if worker:
            if _threading.current_thread().ident == worker.ident:
                _get_logger().warning("Cannot wait on the reactor from its own thread")
                return False
            return wait_thread(worker, timeout)
        return True

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
