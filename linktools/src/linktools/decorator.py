#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import inspect
import threading
from typing import TYPE_CHECKING

from linktools.types import MISSING
from linktools.types import Timeout

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, ParamSpec, TypeVar

    T = TypeVar("T")
    P = ParamSpec("P")
    WRAPPER = Callable[[T], T]


def singleton(cls: "type[T]") -> "Callable[P, T]":
    """Decorate a class so construction returns a single shared instance.

    Returns:
        Callable[P, T]: The operation result.
    """
    instance = MISSING
    lock = threading.RLock()

    @functools.wraps(cls)
    def wrapper(*args: "Any", **kwargs: "Any") -> "T":
        nonlocal instance
        if instance is MISSING:
            with lock:
                if instance is MISSING:
                    instance = cls(*args, **kwargs)
        return instance

    return wrapper


def try_except(
        errors: "tuple[type[BaseException]]" = (Exception,),
        default: "Any" = None,
) -> "Callable[[Callable[P, T]], Callable[P, T]]":
    """Decorate a function to return a default value for selected exceptions.

    Args:
        errors (Tuple[Type[BaseException]]): The errors value.
        default (Any): Value returned when no explicit value is available.

    Returns:
        Any: The operation result.
    """
    def decorator(fn: "Callable[P, T]") -> "Callable[P, T]":
        @functools.wraps(fn)
        def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> "T":
            try:
                return fn(*args, **kwargs)
            except errors:
                return default

        return wrapper

    return decorator


class _CachedProperty:

    def __init__(self, func: "Callable[P, T]", lock: "Any"):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock
        self._probe = threading.local()

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def _get_cached(self, instance: "Any") -> "Any":
        self._probe.active = True
        try:
            return getattr(instance, self.attrname, MISSING)
        finally:
            self._probe.active = False

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")

        if getattr(self._probe, "active", False):
            return MISSING

        val = self._get_cached(instance)
        if val is MISSING:
            if self.lock is not None:
                with self.lock:
                    val = self._get_cached(instance)
                    if val is MISSING:
                        val = self.func(instance)
                        try:
                            setattr(instance, self.attrname, val)
                        except (AttributeError, TypeError):
                            raise TypeError(
                                "instance does not support caching property %r" %
                                (self.attrname,)
                            ) from None
            else:
                val = self.func(instance)
                try:
                    setattr(instance, self.attrname, val)
                except (AttributeError, TypeError):
                    raise TypeError(
                        "instance does not support caching property %r" %
                        (self.attrname,)
                    ) from None

        return val


def cached_property(
        fn: "Callable[P, T]" = None, *, lock: bool = False,
) -> "_CachedProperty | Callable[[Callable[P, T]], _CachedProperty]":
    """Create a property that caches its computed value on the instance.

    Args:
        fn (Callable[P, T]): Callable to invoke.
        lock (bool): The lock value.

    Returns:
        Any: The operation result.
    """
    if fn is not None:
        return _CachedProperty(fn, threading.RLock() if lock else None)

    def decorator(fn: "Callable[P, T]") -> _CachedProperty:
        return _CachedProperty(fn, threading.RLock() if lock else None)

    return decorator


class classproperty:
    """Decorator that converts a method with a single cls argument into a property"""

    def __init__(self, func: "Callable[..., Any] | None" = None) -> None:
        self.func = func

    def __get__(self, instance: "Any", owner: "type | None" = None) -> "Any":
        return self.func(owner)


class _CachedClassproperty:

    def __init__(self, func: "Callable[P, T]", lock: "Any"):
        self.func = func
        self.__doc__ = func.__doc__
        self.lock = lock
        self.val = MISSING

    def __get__(self, instance, owner=None):
        if self.val is MISSING:
            if self.lock is not None:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    if self.val is MISSING:
                        self.val = self.func(owner)
            else:
                self.val = self.func(owner)

        return self.val


def cached_classproperty(
        fn: "Callable[P, T]" = None, *, lock: bool = False
) -> "_CachedClassproperty | Callable[[Callable[P, T]], _CachedClassproperty]":
    """Create a class property that caches its computed value.

    Args:
        fn (Callable[P, T]): Callable to invoke.
        lock (bool): The lock value.

    Returns:
        Union[_CachedClassproperty, Callable[[Callable[P, T]], _CachedClassproperty]]: The operation result.
    """
    if fn is not None:
        return _CachedClassproperty(fn, threading.RLock() if lock else None)

    def decorator(fn: "Callable[P, T]") -> _CachedClassproperty:
        return _CachedClassproperty(fn, threading.RLock() if lock else None)

    return decorator


def _timeoutable(fn: "Callable[P, T]") -> "Callable[P, T]":
    timeout_keyword = "timeout"

    timeout_index = -1
    positional_index = -1
    keyword_index = -1

    index = 0
    for key, parameter in inspect.signature(fn).parameters.items():
        if key == timeout_keyword:
            timeout_index = index
            break
        elif parameter.kind in (parameter.KEYWORD_ONLY, parameter.VAR_KEYWORD):
            keyword_index = index
        elif parameter.kind in (parameter.VAR_POSITIONAL,):
            positional_index = index
        index += 1

    if timeout_index < 0 and keyword_index < 0:
        raise RuntimeError(f"Not found timeout parameter in {fn}")

    if 0 <= positional_index < timeout_index:
        # If timeout appears after *args, it can only be accessed through **kwargs.
        timeout_index = -1

    @functools.wraps(fn)
    def wrapper(*args: "P.args", **kwargs: "P.kwargs") -> "T":
        if 0 <= timeout_index < len(args):
            args = list(args)
            args[timeout_index] = Timeout(args[timeout_index])
        elif timeout_keyword in kwargs:
            kwargs[timeout_keyword] = Timeout(kwargs.get(timeout_keyword))
        else:
            kwargs[timeout_keyword] = Timeout()

        return fn(*args, **kwargs)

    return wrapper


timeoutable: "WRAPPER" = _timeoutable
