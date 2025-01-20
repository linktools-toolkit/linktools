#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : decorator.py
@time    : 2019/01/15
@site    :  
@software: PyCharm 

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
import functools
import inspect
import threading
from typing import TYPE_CHECKING, TypeVar, Type, Any, Callable, Tuple

from .metadata import __missing__
from .types import Timeout

if TYPE_CHECKING:
    from typing import ParamSpec

    T = TypeVar("T")
    P = ParamSpec("P")


def singleton(cls: "Type[T]") -> "Callable[P, T]":
    instance = __missing__
    lock = threading.RLock()

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        nonlocal instance
        if instance is __missing__:
            with lock:
                if instance is __missing__:
                    instance = cls(*args, **kwargs)
        return instance

    return wrapper


def try_except(errors: "Tuple[Type[BaseException]]" = (Exception,), default: "Any" = None):
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

    def __init__(self, func: "Callable[P, T]", lock):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.lock = lock

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it.")

        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None

        val = cache.get(self.attrname, __missing__)
        if val is __missing__:
            if self.lock is not None:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    val = cache.get(self.attrname, __missing__)
                    if val is __missing__:
                        val = self.func(instance)
                        try:
                            cache[self.attrname] = val
                        except TypeError:
                            msg = (
                                f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                                f"does not support item assignment for caching {self.attrname!r} property."
                            )
                            raise TypeError(msg) from None
            else:
                val = self.func(instance)
                try:
                    cache[self.attrname] = val
                except TypeError:
                    msg = (
                        f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                        f"does not support item assignment for caching {self.attrname!r} property."
                    )
                    raise TypeError(msg) from None

        return val


def cached_property(fn: "Callable[P, T]" = None, *, lock: bool = False):
    if fn is not None:
        return _CachedProperty(fn, threading.RLock() if lock else None)

    def decorator(fn: "Callable[P, T]"):
        return _CachedProperty(fn, threading.RLock() if lock else None)

    return decorator


class classproperty:
    """
    Decorator that converts a method with a single cls argument into a property
    that can be accessed directly from the class.
    """

    def __init__(self, func=None):
        self.func = func

    def __get__(self, instance, owner=None):
        return self.func(owner)


class _CachedClassproperty:

    def __init__(self, func: "Callable[P, T]", lock):
        self.func = func
        self.__doc__ = func.__doc__
        self.lock = lock
        self.val = __missing__

    def __get__(self, instance, owner=None):
        if self.val is __missing__:
            if self.lock is not None:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    if self.val is __missing__:
                        self.val = self.func(owner)
            else:
                self.val = self.func(owner)

        return self.val


def cached_classproperty(fn: "Callable[P, T]" = None, *, lock: bool = False):
    if fn is not None:
        return _CachedClassproperty(fn, threading.RLock() if lock else None)

    def decorator(fn: "Callable[P, T]"):
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
        # 如果timeout在*args参数后面，那就只能通过**kwargs访问了
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


timeoutable: "Callable[[T], T]" = _timeoutable
