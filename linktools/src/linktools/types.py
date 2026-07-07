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
import time as _time
import types as _types
import typing as _t
import weakref as _weakref
from pathlib import Path as _Path


# Monotonic clock for in-process deadlines. Spec §3.6/§6.2: wall-clock changes
# (NTP, DST, manual) must never affect timeout/scheduling correctness; persistent
# TTLs use UTC unix timestamps via time.time() elsewhere.
_now = _time.monotonic


class __MissingType:
    __eq__ = lambda l, r: \
        l is r or type(l) is type(r)
    __bool__ = lambda _: False
    __repr__ = lambda _: "<MISSING>"


MISSING = __MissingType()
MissingType = __MissingType

T = _t.TypeVar("T")
PathType = _t.Union[str, _Path]
QueryDataType = _t.Union[str, int, float]
QueryType = _t.Dict[str, _t.Union[QueryDataType, _t.List[QueryDataType], _t.Tuple[QueryDataType, ...]]]
TimeoutType = _t.Union["Timeout", float, int, None]


if _t.TYPE_CHECKING:
    from .core._config import ConfigDict, Config, ConfigKeyType, ConfigLiteralType, ConfigType, ConfigTypeMap  # noqa
    from .core._tools import Tools, Tool, ToolExecError  # noqa
    from .core._url import UrlFile, UrlFileValidatorType  # noqa
    from .core._environ import BaseEnviron as _BaseEnviron  # noqa

    P = _t.ParamSpec("P")
    EnvironType = _t.TypeVar("EnvironType", bound=_BaseEnviron)


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
    """Track a monotonic deadline for an in-process operation.

    Uses :func:`time.monotonic` so wall-clock adjustments (NTP jumps, DST,
    manual changes) never affect correctness -- see spec §3.6 and §6.2.

    Semantics:
    * ``None`` means wait forever (infinite).
    * ``0`` means the deadline has already passed.
    * Negative values are rejected with ``ValueError``.
    * :attr:`remaining` / :attr:`expired` are the canonical reads;
      :meth:`check` / :meth:`ensure` are retained as conveniences.
    """

    def __new__(cls, timeout: "TimeoutType" = None):
        if isinstance(timeout, cls):
            return timeout
        if timeout is None or isinstance(timeout, (float, int)):
            if timeout is not None and timeout < 0:
                raise ValueError("timeout must be non-negative, got %r" % (timeout,))
            t = super().__new__(cls)
            t._timeout = timeout
            t._deadline = None
            t.reset()
            return t
        raise TypeError("Timeout/int/float/None expected, got %s" % type(timeout).__name__)

    @property
    def timeout(self) -> "_t.Optional[float]":
        """Configured duration in seconds, or ``None`` for infinite."""
        return self._timeout

    @property
    def deadline(self) -> "_t.Optional[float]":
        """Monotonic deadline, or ``None`` when the timeout is infinite."""
        return self._deadline

    @property
    def remaining(self) -> "_t.Optional[float]":
        """Seconds left until the deadline, clamped at 0; ``None`` if infinite."""
        if self._deadline is None:
            return None
        return max(self._deadline - _now(), 0)

    @property
    def expired(self) -> bool:
        """``True`` once the deadline has passed; never ``True`` when infinite."""
        return self._deadline is not None and _now() >= self._deadline

    def check(self) -> bool:
        """Return ``True`` while time remains (the inverse of :attr:`expired`)."""
        return not self.expired

    def reset(self) -> None:
        """Recompute the deadline from the current monotonic time."""
        if self._timeout is not None:
            self._deadline = _now() + self._timeout
        else:
            self._deadline = None

    def ensure(self, err_type: type = TimeoutError, message: str = "Timeout") -> None:
        """Raise ``err_type(message)`` if the deadline has passed."""
        if self.expired:
            raise err_type(message)

    def split(self, timeout: "TimeoutType" = None) -> "Timeout":
        """Return a child timeout bounded by both ``timeout`` and our budget.

        The child never outlives the parent: its deadline is the sooner of
        ``now + timeout`` and this timeout's own deadline. ``timeout=None``
        yields a child that shares the parent's remaining (possibly infinite)
        budget. Negative ``timeout`` is rejected by :meth:`Timeout.__new__`.
        """
        if isinstance(timeout, Timeout):
            timeout = timeout._timeout
        if self._deadline is None:
            # Parent is infinite: child is governed only by the requested value.
            return Timeout(timeout)
        remaining = self.remaining
        if timeout is None:
            return Timeout(remaining)
        return Timeout(min(timeout, remaining))

    def __repr__(self) -> str:
        return "Timeout(timeout=%r)" % (self._timeout,)


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
