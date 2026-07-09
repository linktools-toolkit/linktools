#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : utils.py
@time    : 2018/11/25
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
import gzip
import subprocess
from typing import TYPE_CHECKING

from ..types import MISSING

if TYPE_CHECKING:
    import logging

    from collections.abc import Callable, Iterable
    from typing import Any, ParamSpec, TypeVar

    from ..core import Environ

    T = TypeVar("T")
    R = TypeVar("R")
    P = ParamSpec("P")

_environ = _logger = None


def get_environ() -> "Environ":
    """Return the active linktools environment object.

    Returns:
        Environ: The operation result.
    """
    global _environ
    if _environ is None:
        from ..core import environ
        _environ = environ
    return _environ


def get_logger() -> "logging.Logger":
    """Return the logger for the active linktools environment.

    Returns:
        logging.Logger: The operation result.
    """
    global _logger
    if _logger is None:
        _logger = get_environ().get_logger("utils")
    return _logger


def ignore_errors(
        fn: "Callable[P, T]", *,
        args: "P.args" = None, kwargs: "P.kwargs" = None,
        default: "T" = None
) -> "T":
    """Run a callable and suppress the selected exception types.

    Args:
        fn (Callable[P, T]): Callable to invoke.
        args (P.args): Arguments passed to the operation.
        kwargs (P.kwargs): Keyword arguments passed to the operation.
        default (T): Value returned when no explicit value is available.

    Returns:
        T: The operation result.
    """
    try:
        if args is not None:
            return fn(*args, **kwargs) \
                if kwargs is not None \
                else fn(*args)
        else:
            return fn(**kwargs) \
                if kwargs is not None \
                else fn()
    except Exception:
        return default


def cast(type: "type[T]", obj: "Any", default: "Any" = MISSING) -> "T | None":  # noqa
    """Cast a value to the requested type.

    Args:
        type (Type[T]): Target type used to cast the value.
        obj (Any): Object to inspect or convert.
        default (Any): Value returned when no explicit value is available.

    Returns:
        Optional[T]: The operation result.
    """
    if default is MISSING:
        return type(obj)
    try:
        return type(obj)
    except Exception:
        return default


def cast_int(obj: "Any", default: "Any" = MISSING) -> int:
    """Cast a value to int.

    Args:
        obj (Any): Object to inspect or convert.
        default (Any): Value returned when no explicit value is available.

    Returns:
        int: The operation result.
    """
    return cast(int, obj, default)


def cast_bool(obj: "Any", default: "Any" = MISSING) -> bool:
    """Cast a value to bool.

    Args:
        obj (Any): Object to inspect or convert.
        default (Any): Value returned when no explicit value is available.

    Returns:
        bool: The operation result.
    """
    return cast(bool, obj, default)


def coalesce(*args: "Any") -> "Any":
    """Return the first non-None value from the arguments.

    Args:
        args (Any): Arguments passed to the operation.

    Returns:
        Any: The operation result.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def is_contain(obj: "Any", key: "Any") -> bool:
    """Return whether an object contains a key.

    Args:
        obj (Any): Object to inspect or convert.
        key (Any): Configuration or item key.

    Returns:
        bool: The operation result.
    """
    if obj is None:
        return False
    if hasattr(obj, "__contains__"):
        return key in obj
    return False


def is_empty(obj: "Any") -> bool:
    """Return whether an object is empty.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        bool: The operation result.
    """
    if obj is None:
        return True
    if hasattr(obj, "__len__"):
        return len(obj) == 0
    return False


def get_item(obj: "Any", *keys: "Any", type: "type[T]" = None, default: "T" = None) -> "T | None":  # noqa
    """Return a nested item or attribute from an object.

    Args:
        obj (Any): Object to inspect or convert.
        keys (Any): Keys to inspect or update.
        type (Type[T]): Target type used to cast the value.
        default (T): Value returned when no explicit value is available.

    Returns:
        Optional[T]: The operation result.
    """
    for key in keys:
        if obj is None:
            return default

        try:
            obj = obj[key]
            continue
        except Exception:
            pass

        try:
            obj = getattr(obj, key)
            continue
        except Exception:
            pass

        return default

    if obj is not None and type is not None:
        try:
            obj = type(obj)
        except Exception:
            return default

    return obj


def pop_item(obj: "Any", *keys: "Any", type: "type[T]" = None, default: "T" = None) -> "T | None":  # noqa
    """Return and remove a nested item from an object.

    Args:
        obj (Any): Object to inspect or convert.
        keys (Any): Keys to inspect or update.
        type (Type[T]): Target type used to cast the value.
        default (T): Value returned when no explicit value is available.

    Returns:
        Optional[T]: The operation result.
    """
    last_obj = None
    last_key = None

    for key in keys:

        if obj is None:
            return default

        last_obj = obj
        last_key = key

        try:
            obj = obj[key]
            continue
        except Exception:
            pass

        try:
            obj = getattr(obj, key)
            continue
        except Exception:
            pass

        return default

    if last_obj is not None and last_key is not None:
        try:
            del last_obj[last_key]
        except Exception:
            pass

    if obj is not None and type is not None:
        try:
            obj = type(obj)
        except Exception:
            return default

    return obj


def get_list_item(obj: "Any", *keys: "Any", type: "type[T]" = None, default: "list[T]" = None) -> "list[T] | None":  # noqa
    """Return a list item after trying several indexes.

    Args:
        obj (Any): Object to inspect or convert.
        keys (Any): Keys to inspect or update.
        type (Type[T]): Target type used to cast the value.
        default (List[T]): Value returned when no explicit value is available.

    Returns:
        Optional[List[T]]: The operation result.
    """
    objs = get_item(obj, *keys, default=None)
    if objs is None or not isinstance(objs, (tuple, list, set)):
        return default
    result = []
    for obj in objs:
        if obj is not None and type is not None:
            try:
                result.append(type(obj))
            except Exception:
                pass
        else:
            result.append(obj)
    return result


def make_uuid() -> str:
    """Return a random UUID string.

    Returns:
        str: The operation result.
    """
    import random
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{uuid.uuid1()}{random.random()}")).replace("-", "")


def random_string(length: int = 16) -> str:
    """Return a random string using the requested character set.

    Args:
        length (int): The length value.

    Returns:
        str: The operation result.
    """
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def gzip_compress(data: "str | bytes") -> bytes:
    """Return gzip-compressed bytes for the supplied data.

    Args:
        data (Union[str, bytes]): The data value.

    Returns:
        bytes: The operation result.
    """
    if isinstance(data, str):
        data = bytes(data, "utf8")
    return gzip.compress(data)


def parse_version(version: str) -> "tuple[int, ...]":
    """Parse a version string into a comparable tuple.

    Args:
        version (str): The version value.

    Returns:
        Tuple[int, ...]: The operation result.
    """
    result = []
    for x in version.split("."):
        if x.isdigit():
            result.append(cast_int(x))
        else:
            import re
            match = re.match(r"^\d+", x)
            if not match:
                break
            result.append(cast_int(match.group(0)))
    return tuple(result)


_widths = [
    (126, 1), (159, 0), (687, 1), (710, 0), (711, 1),
    (727, 0), (733, 1), (879, 0), (1154, 1), (1161, 0),
    (4347, 1), (4447, 2), (7467, 1), (7521, 0), (8369, 1),
    (8426, 0), (9000, 1), (9002, 2), (11021, 1), (12350, 2),
    (12351, 1), (12438, 2), (12442, 0), (19893, 2), (19967, 1),
    (55203, 2), (63743, 1), (64106, 2), (65039, 1), (65059, 0),
    (65131, 2), (65279, 1), (65376, 2), (65500, 1), (65510, 2),
    (120831, 1), (262141, 2), (1114109, 1),
]


def get_char_width(char):
    """Return the display width of a character.

    Args:
        char: The char value.

    Returns:
        Any: The operation result.
    """
    global _widths
    o = ord(char)
    if o == 0xe or o == 0xf:
        return 0
    for num, wid in _widths:
        if o <= num:
            return wid
    return 1


def let(value: "T", fn: "Callable[[T], R]") -> "R":
    """Apply a function to a value and return the function result.

    Args:
        value (T): Value to store or process.
        fn (Callable[[T], R]): Callable to invoke.

    Returns:
        R: The operation result.
    """
    return fn(value)


def also(value: "T", fn: "Callable[[T], Any]") -> "T":
    """Apply a function to a value and return the original value.

    Args:
        value (T): Value to store or process.
        fn (Callable[[T], Any]): Callable to invoke.

    Returns:
        T: The operation result.
    """
    fn(value)
    return value


def list2cmdline(args: "Iterable[str]") -> str:
    return subprocess.list2cmdline(args)


def cmdline2list(cmdline: str) -> "Iterable[str]":
    import shlex
    return shlex.split(cmdline)
