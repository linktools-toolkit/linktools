#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Event dispatch (spec §15.2).

:class:`EventBus` is the canonical dispatcher: ``on`` returns a cancellable
subscription, ``emit`` honors a configurable exception policy, and callbacks run
against a snapshot outside the registration lock (so a callback may cancel
itself or register new handlers without deadlocking).
"""

import threading as _threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

# Exception policies (spec  RUN-EVT-002).
LOG_AND_CONTINUE = "log_and_continue"  # default; CLI-friendly
RAISE_FIRST = "raise_first"
COLLECT = "collect"
STOP = "stop"
_ALL_POLICIES = (LOG_AND_CONTINUE, RAISE_FIRST, COLLECT, STOP)

_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        from ..core import environ
        _logger = environ.get_logger("events")
    return _logger


class Subscription(object):
    """A cancellable event subscription returned by :meth:`EventBus.on`."""

    def __init__(self, bus, event, callback):
        self._bus = bus
        self._event = event
        self._callback = callback
        self._cancelled = False

    @property
    def cancelled(self):
        return self._cancelled

    def cancel(self) -> None:
        """Remove this subscription. Idempotent."""
        if self._cancelled:
            return
        self._cancelled = True
        self._bus._remove(self._event, self._callback)


class EventBus(object):
    """A thread-safe named-event dispatcher (spec §15.2)."""

    def __init__(self, exception_policy: str = LOG_AND_CONTINUE) -> None:
        if exception_policy not in _ALL_POLICIES:
            raise ValueError("unknown exception policy: %r" % (exception_policy,))
        self._policy = exception_policy
        self._lock = _threading.RLock()
        # event -> {callback: {"time": int, "max_times": Optional[int]}}
        self._handlers: "dict[str, dict]" = {}

    # -- registration ------------------------------------------------------

    def on(self, event: str, callback: "Callable", times: "int | None" = None) -> "Subscription":
        with self._lock:
            callbacks = self._handlers.setdefault(event, {})
            callbacks[callback] = {"time": 0, "max_times": times}
        return Subscription(self, event, callback)

    def once(self, event: str, callback: "Callable") -> "Subscription":
        return self.on(event, callback, times=1)

    def off(self, event: str, callback: "Callable") -> None:
        self._remove(event, callback)

    def _remove(self, event, callback):
        with self._lock:
            callbacks = self._handlers.get(event)
            if callbacks is None:
                return
            callbacks.pop(callback, None)
            if not callbacks:
                self._handlers.pop(event, None)

    # -- dispatch ----------------------------------------------------------

    def emit(self, event: str, *args, **kwargs) -> None:
        logger = _get_logger()
        # Snapshot under the lock; invoke outside it so callbacks may
        # cancel themselves or register new handlers ( RUN-EVT-004).
        with self._lock:
            callbacks = self._handlers.get(event)
            if not callbacks:
                return
            invoke = []
            remove = []
            for callback, info in callbacks.items():
                invoke.append(callback)
                info["time"] += 1
                if info["max_times"] is not None and info["time"] >= info["max_times"]:
                    remove.append(callback)
            for callback in remove:
                callbacks.pop(callback, None)
            if not callbacks:
                self._handlers.pop(event, None)

        logger.debug("Event `%s` invoke %d callbacks" % (event, len(invoke)))
        collected = []
        for callback in invoke:
            try:
                callback(*args, **kwargs)
            except Exception as e:  #  RUN-EVT-002
                if self._policy == LOG_AND_CONTINUE:
                    logger.warning("Event `%s` handler `%s` error" % (event, callback), exc_info=e)
                elif self._policy == RAISE_FIRST:
                    raise
                elif self._policy == STOP:
                    raise
                elif self._policy == COLLECT:
                    collected.append(e)
        if collected:
            raise collected[0]
