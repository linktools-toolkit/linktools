#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : types.py
@time    : 2024/7/21
@site    : https://github.com/ice-black-tea
@software: PyCharm
"""

import abc as _abc
import logging as _logging
import threading as _threading
import time as _time
import types as _types
import typing as _t
import weakref as _weakref
from pathlib import Path as _Path

T = _t.TypeVar("T")
PathType = _t.Union[str, _Path]
QueryDataType = _t.Union[str, int, float]
QueryType = _t.Dict[str, _t.Union[QueryDataType, _t.List[QueryDataType], _t.Tuple[QueryDataType, ...]]]
TimeoutType = _t.Union["Timeout", float, int, None]

if _t.TYPE_CHECKING:
    from .core._config import ConfigDict, Config, ConfigKeyType, ConfigLiteralType, ConfigType, ConfigTypeMap  # noqa
    from .core._tools import Tools, Tool, ToolExecError  # noqa
    from .core._url import UrlFile, UrlFileValidatorType  # noqa
    from .core import BaseEnviron as _BaseEnviron  # noqa

    P = _t.ParamSpec("P")
    EnvironType = _t.TypeVar("EnvironType", bound=_BaseEnviron)

_logger: "_logging.Logger | None" = None


def _get_logger() -> "_logging.Logger":
    global _logger
    if _logger is None:
        from .core import environ
        _logger = environ.get_logger()
    return _logger


def get_origin(tp):
    """Return the unsubscripted origin for a typing object."""
    if hasattr(_t, "get_origin"):
        return _t.get_origin(tp)
    if tp is _t.Generic:
        return _t.Generic
    union_type = getattr(_types, "UnionType", None)
    if union_type is not None and isinstance(tp, union_type):
        return union_type
    if hasattr(tp, "__origin__"):
        return tp.__origin__
    raise TypeError(f"{tp} has no attribute '__origin__'")


def get_args(tp):
    """Return the type arguments for a typing object."""
    if hasattr(_t, "get_args"):
        return _t.get_args(tp)
    if hasattr(tp, "__args__"):
        return tp.__args__
    raise TypeError(f"{tp} has no attribute '__args__'")


class Timeout:
    """Track timeout state and compute remaining time for operations."""

    _timeout: "float | None"
    _deadline: "float | None"

    def __new__(cls, timeout: "TimeoutType" = None):
        if isinstance(timeout, cls):
            return timeout
        if isinstance(timeout, (float, int, type(None))):
            t = super().__new__(cls)
            t._timeout = timeout
            t._deadline = None
            t.reset()
            return t
        raise TypeError(f"Timeout/int/float was expects, got {type(timeout)}")

    @property
    def remain(self) -> "float | None":
        timeout = None
        if self._deadline is not None:
            timeout = max(self._deadline - _time.time(), 0)
        return timeout

    @property
    def deadline(self) -> "float | None":
        return self._deadline

    def reset(self) -> None:
        if self._timeout is not None and self._timeout >= 0:
            self._deadline = _time.time() + self._timeout

    def check(self) -> bool:
        if self._deadline is not None and _time.time() > self._deadline:
            return False
        return True

    def ensure(self, err_type: "_t.Callable[[str], Exception]" = TimeoutError, message: str = "Timeout") -> None:
        if not self.check():
            raise err_type(message)

    def __repr__(self):
        return f"Timeout(timeout={self._timeout})"


class Stoppable(_abc.ABC):
    """Stoppable interface"""

    @_abc.abstractmethod
    def stop(self):
        """Stop the running resource or background operation."""
        pass

    def _stop_on_error(self, callback: "_t.Callable[P, T]", *args: "P.args", **kwargs: "P.kwargs") -> "T":
        try:
            return callback(*args, **kwargs)
        except:
            self.stop()
            raise

    def _stop_on_exit(self):
        _weakref.finalize(self, self.stop)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()


class _EventHandler(dict):

    def __init__(self):
        super().__init__()
        self._lock = _threading.RLock()

    @property
    def lock(self) -> "_threading.RLock":
        return self._lock


_event_handler_lock = _threading.RLock()
_event_handler_name = "__EventHandlerMixin_event_handler"


class EventHandlerMixin(object):
    """Dispatch named events to registered handlers."""

    @property
    def _event_handler(self) -> "_EventHandler":
        value = getattr(self, _event_handler_name, None)
        if value is None:
            with _event_handler_lock:
                value = getattr(self, _event_handler_name, None)
                if value is None:
                    value = _EventHandler()
                    setattr(self, _event_handler_name, value)
        return value

    def on(self, event: str, callback: "_t.Callable[..., _t.Any]", times: int = None):
        logger = _get_logger()
        handler = self._event_handler
        with handler.lock:
            logger.debug(f"Register event `{event}` handler `{callback}`")
            callbacks = handler.get(event, None)
            if callbacks is None:
                callbacks = handler[event] = dict()
            callbacks[callback] = {
                "time": 0,
                "max_times": times,
            }

    def off(self, event: str, callback: "_t.Callable[..., _t.Any]"):
        logger = _get_logger()
        handler = self._event_handler
        with handler.lock:
            logger.debug(f"Unregister event `{event}` handler `{callback}`")
            if event in handler:
                callbacks = handler.get(event)
                try:
                    callbacks.pop(callback)
                except KeyError:
                    pass

    def once(self, event: str, callback: "_t.Callable[..., _t.Any]"):
        self.on(event, callback, 1)

    def trigger(self, event: str, *args: "_t.Any", **kwargs: "_t.Any"):
        logger = _get_logger()
        handler = self._event_handler
        invoke_list, remove_list = [], []
        with handler.lock:
            if event in handler:
                callbacks = handler.get(event)
                for callback, info in callbacks.items():
                    invoke_list.append(callback)
                    info["time"] += 1
                    if info["max_times"] is not None and info["time"] >= info["max_times"]:
                        remove_list.append(callback)
            for callback in remove_list:
                callbacks.pop(callback)
        logger.debug(f"Event `{event}` invoke {len(invoke_list)} callbacks")
        for callback in invoke_list:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Event `{event}` handler `{callback}` error", exc_info=e)
