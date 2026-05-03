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
import functools
import gzip
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, overload, Tuple, List, Set

from ..decorator import timeoutable
from ..metadata import __missing__
from ..types import Proxy, Error

if TYPE_CHECKING:
    import subprocess
    import threading
    import logging

    from collections.abc import Callable
    from typing import Any, Literal, ParamSpec, TypeVar
    from importlib.machinery import ModuleSpec

    from ..core import Environ
    from ..types import PathType, QueryType, TimeoutType

    T = TypeVar("T")
    R = TypeVar("R")
    P = ParamSpec("P")

DEFAULT_ENCODING = "utf-8"

_is_windows_like = _is_unix_like = False

try:
    import msvcrt
except ModuleNotFoundError:
    try:
        import pwd
    except ModuleNotFoundError:
        ...
    else:
        _is_unix_like = True
else:
    _is_windows_like = True

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
    except:
        return default


def cast(type: "type[T]", obj: "Any", default: "Any" = __missing__) -> "T | None":  # noqa
    """Cast a value to the requested type.

    Args:
        type (Type[T]): Target type used to cast the value.
        obj (Any): Object to inspect or convert.
        default (Any): Value returned when no explicit value is available.

    Returns:
        Optional[T]: The operation result.
    """
    if default is __missing__:
        return type(obj)
    try:
        return type(obj)
    except:
        return default


def cast_int(obj: "Any", default: "Any" = __missing__) -> int:
    """Cast a value to int.

    Args:
        obj (Any): Object to inspect or convert.
        default (Any): Value returned when no explicit value is available.

    Returns:
        int: The operation result.
    """
    return cast(int, obj, default)


def cast_bool(obj: "Any", default: "Any" = __missing__) -> bool:
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
        except:
            pass

        try:
            obj = getattr(obj, key)
            continue
        except:
            pass

        return default

    if obj is not None and type is not None:
        try:
            obj = type(obj)
        except:
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
        except:
            pass

        try:
            obj = getattr(obj, key)
            continue
        except:
            pass

        return default

    if last_obj is not None and last_key is not None:
        try:
            del last_obj[last_key]
        except:
            pass

    if obj is not None and type is not None:
        try:
            obj = type(obj)
        except:
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
    if objs is None or not isinstance(objs, (Tuple, List, Set)):
        return default
    result = []
    for obj in objs:
        if obj is not None and type is not None:
            try:
                result.append(type(obj))
            except:
                pass
        else:
            result.append(obj)
    return result


def get_hash(data: "str | bytes", algorithm: "Literal['md5', 'sha1', 'sha256']" = "md5") -> str:
    """Return the digest for bytes or text using the selected hash algorithm.

    Args:
        data (Union[str, bytes]): The data value.
        algorithm ("Literal['md5', 'sha1', 'sha256']"): The algorithm value.

    Returns:
        str: The operation result.
    """
    import hashlib
    if isinstance(data, str):
        data = bytes(data, "utf8")
    m = getattr(hashlib, algorithm)()
    m.update(data)
    return m.hexdigest()


def get_file_hash(path: "PathType", algorithm: "Literal['md5', 'sha1', 'sha256']" = "md5") -> str:
    """Return the digest for a file using the selected hash algorithm.

    Args:
        path (PathType): Filesystem path to process.
        algorithm ("Literal['md5', 'sha1', 'sha256']"): The algorithm value.

    Returns:
        str: The operation result.
    """
    import hashlib
    m = getattr(hashlib, algorithm)()
    with open(path, "rb") as fd:
        while True:
            data = fd.read(4096 << 4)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def get_md5(data: "str | bytes") -> str:
    """Return the MD5 digest for bytes or text.

    Args:
        data (Union[str, bytes]): The data value.

    Returns:
        str: The operation result.
    """
    return get_hash(data, algorithm="md5")


def get_file_md5(path: "PathType"):
    """Return the MD5 digest for a file.

    Args:
        path (PathType): Filesystem path to process.

    Returns:
        Any: The operation result.
    """
    return get_file_hash(path, algorithm="md5")


def get_hash_ident(data: "str | bytes"):
    """Return a short stable identifier from a hashed value.

    Args:
        data (Union[str, bytes]): The data value.

    Returns:
        Any: The operation result.
    """
    if isinstance(data, str):
        data = bytes(data, "utf8")
    length = f"{len(data):0>4x}"
    md5 = get_hash(data, "md5")
    sha1 = get_hash(data, "sha1")
    return f"{length[-4:]}{md5[:6]}{sha1[:6]}"


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


def is_sub_path(path: "PathType", root_path: "PathType") -> bool:
    """Return whether a path is contained within another path.

    Args:
        path (PathType): Filesystem path to process.
        root_path (PathType): The root_path value.

    Returns:
        bool: The operation result.
    """
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        abs_root_path = os.path.abspath(os.path.expanduser(root_path))
        return os.path.commonpath([abs_path, abs_root_path]) == abs_root_path
    except ValueError:
        return False


def join_path(root_path: "PathType", *paths: str) -> "Path":
    """Join path segments and optionally expand user or environment markers.

    Args:
        root_path (PathType): The root_path value.
        paths (str): The paths value.

    Returns:
        Path: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    target_path = str(root_path)
    for path in paths:
        parent_path = target_path
        target_path = os.path.abspath(os.path.join(target_path, path))
        try:
            if os.path.commonpath([target_path, parent_path]) != parent_path:
                raise Error(f"Unsafe path \"{path}\"")
        except ValueError:
            raise Error(f"Unsafe path \"{path}\"")
    return Path(target_path)


@overload
def read_file(path: "PathType") -> bytes:
    """Read file data from a path.

    Args:
        path (PathType): Filesystem path to process.

    Returns:
        bytes: The operation result.
    """
    ...


@overload
def read_file(path: "PathType", text: "Literal[False]") -> bytes:
    """Read file data from a path.

    Args:
        path (PathType): Filesystem path to process.
        text (Literal[False]): Whether file data should be decoded as text.

    Returns:
        bytes: The operation result.
    """
    ...


@overload
def read_file(path: "PathType", text: "Literal[True]", encoding: str = DEFAULT_ENCODING) -> str:
    """Read file data from a path.

    Args:
        path (PathType): Filesystem path to process.
        text (Literal[True]): Whether file data should be decoded as text.
        encoding (str): Text encoding used for file data.

    Returns:
        str: The operation result.
    """
    ...


@overload
def read_file(path: "PathType", text: bool, encoding: str = DEFAULT_ENCODING) -> "str | bytes":
    """Read file data from a path.

    Args:
        path (PathType): Filesystem path to process.
        text (bool): Whether file data should be decoded as text.
        encoding (str): Text encoding used for file data.

    Returns:
        Union[str, bytes]: The operation result.
    """
    ...


def read_file(path: "PathType", text: bool = False, encoding: str = DEFAULT_ENCODING) -> "str | bytes":
    """Read data from a file.

    Args:
        path (PathType): Filesystem path to process.
        text (bool): Whether file data should be decoded as text.
        encoding (str): Text encoding used for file data.

    Returns:
        Union[str, bytes]: The operation result.
    """
    if text:
        with open(path, "rt", encoding=encoding) as fd:
            return fd.read()
    else:
        with open(path, "rb") as fd:
            return fd.read()


def write_file(path: "PathType", data: "str | bytes", encoding: str = DEFAULT_ENCODING) -> None:
    """Write data to a file.

    Args:
        path (PathType): Filesystem path to process.
        data (Union[str, bytes]): The data value.
        encoding (str): Text encoding used for file data.
    """
    if isinstance(data, str):
        with open(path, "wt", encoding=encoding) as fd:
            fd.write(data)
    else:
        with open(path, "wb") as fd:
            fd.write(data)


def remove_file(path: "PathType") -> None:
    """Remove a file or directory.

    Args:
        path (PathType): Filesystem path to process.
    """
    if not os.path.exists(path):
        return
    if get_environ().debug:
        get_logger().debug(f"Remove File: {path}")
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    else:
        ignore_errors(os.remove, args=(path,))


def clear_directory(path: "PathType") -> None:
    """Remove child paths from a directory.

    Args:
        path (PathType): Filesystem path to process.
    """
    if not os.path.isdir(path):
        return
    if get_environ().debug:
        get_logger().debug(f"Clear Directory: {path}")
    for name in os.listdir(path):
        target_path = os.path.join(path, name)
        if os.path.isdir(target_path):
            import shutil
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            ignore_errors(os.remove, args=(target_path,))


def get_lan_ip() -> "str | None":
    """Return the local LAN IP address.

    Returns:
        Optional[str]: The operation result.
    """
    s = None
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except:
        return None
    finally:
        if s is not None:
            ignore_errors(s.close)


def get_wan_ip() -> "str | None":
    """Return the public WAN IP address.

    Returns:
        Optional[str]: The operation result.
    """
    from urllib.request import urlopen
    try:
        with urlopen(get_environ().get_config("DEFAULT_WAN_IP_URL")) as response:
            return response.read().decode().strip()
    except:
        return None


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


_user_agent = None


def user_agent(style=None) -> str:
    """Return a random user-agent string.

    Args:
        style: The style value.

    Returns:
        str: The operation result.
    """
    global _user_agent

    if _user_agent is None:
        from linktools.references.fake_useragent import UserAgent

        class _UserAgent(UserAgent):

            def __init__(self):
                environ = get_environ()
                super().__init__(
                    path=environ.get_path("assets", "browsers.json"),
                    fallback=environ.get_config("DEFAULT_USER_AGENT", type=str),
                )

        _user_agent = _UserAgent()

    try:
        if style:
            return _user_agent[style]

        return _user_agent.random

    except Exception as e:
        logger = get_logger()
        logger.debug(f"fetch user agent error: {e}")

    return _user_agent.fallback


def make_url(scheme: str, host: str, port: int, *paths: str, **kwargs: "QueryType") -> str:
    """Build a URL from parts and query values.

    Args:
        scheme (str): The scheme value.
        host (str): The host value.
        port (int): Remote port number.
        paths (str): The paths value.
        kwargs (QueryType): Keyword arguments passed to the operation.

    Returns:
        str: The operation result.
    """
    url = f"{scheme}://{host}"
    if port is not None:
        if (scheme == "http" and port != 80) or (scheme == "https" and port != 443):
            url += f":{port}"
    return join_url(url, *paths, **kwargs)


def join_url(url: str, *paths: str, **kwargs: "QueryType") -> str:
    """Join URL path segments safely.

    Args:
        url (str): URL to process.
        paths (str): The paths value.
        kwargs (QueryType): Keyword arguments passed to the operation.

    Returns:
        str: The operation result.
    """
    from urllib import parse

    result = url

    for path in paths:
        if path:
            result = result.rstrip("/") + "/" + path.lstrip("/")

    if len(kwargs) > 0:
        queries = []
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                queries.extend((key, v) for v in value)
            else:
                queries.append((key, value))

        result = result + "?" + parse.urlencode(queries)

    return result


def guess_file_name(url: str) -> str:
    """Guess a filename from a URL and response metadata.

    Args:
        url (str): URL to process.

    Returns:
        str: The operation result.
    """
    from urllib import parse
    if not url:
        return ""
    try:
        return os.path.split(parse.urlparse(url).path)[1]
    except:
        return ""


def _parseparam(s):
    while s[:1] == ';':
        s = s[1:]
        end = s.find(";")
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(';', end + 1)
        if end < 0:
            end = len(s)
        f = s[:end]
        yield f.strip()
        s = s[end:]


def parse_header(line):
    """Parse a Content-type like header.

    Args:
        line: The line value.

    Returns:
        Any: The operation result.
    """
    parts = _parseparam(';' + line)
    key = parts.__next__()
    pdict = {}
    for p in parts:
        i = p.find("=")
        if i >= 0:
            name = p[:i].strip().lower()
            value = p[i + 1:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
                value = value.replace('\\\\', '\\').replace('\\"', '"')
            pdict[name] = value
    return key, pdict


def parser_cookie(cookie: str) -> "dict[str, str]":
    """Parse a cookie header into a dictionary.

    Args:
        cookie (str): The cookie value.

    Returns:
        Dict[str, str]: The operation result.
    """
    cookies = {}
    for item in cookie.split(";"):
        key_value = item.split("=", 1)
        cookies[key_value[0].strip()] = key_value[1].strip() if len(key_value) > 1 else ''
    return cookies


_interpreter = _interpreter_ident = None


def get_interpreter():
    """Return the absolute path to the current Python interpreter.

    Returns:
        Any: The operation result.
    """
    global _interpreter
    if _interpreter is None:
        _interpreter = sys.executable
    return _interpreter


def get_interpreter_ident() -> str:
    """Return a stable identifier for the current Python interpreter.

    Returns:
        str: The operation result.
    """
    global _interpreter_ident
    if _interpreter_ident is None:
        import platform
        _interpreter_ident = f"{get_hash_ident(sys.exec_prefix)}_{platform.python_version()}"
    return _interpreter_ident


_system = _machine = None


def get_system() -> str:
    """Return the current operating system name.

    Returns:
        str: The operation result.
    """
    global _system
    if _system is None:
        import platform
        _system = platform.system().lower()
    return _system


def get_machine() -> str:
    """Return the current machine architecture name.

    Returns:
        str: The operation result.
    """
    global _machine
    if _machine is None:
        import platform
        _machine = platform.machine().lower()
    return _machine


def is_unix_like(system: str = None) -> bool:
    """Return whether the current system is Unix-like.

    Args:
        system (str): The system value.

    Returns:
        bool: The operation result.
    """
    return system in ("darwin", "linux") if system else _is_unix_like


def is_windows(system: str = None) -> bool:
    """Return whether the current system is Windows.

    Args:
        system (str): The system value.

    Returns:
        bool: The operation result.
    """
    return system == "windows" if system else _is_windows_like


if is_windows():

    def get_user(uid: int = None):
        """Return the user name for a UID or the current user.

        Args:
            uid (int): User id to resolve. If omitted, the current user is used.

        Returns:
            Any: The operation result.
        """
        import getpass
        return getpass.getuser()


    def get_uid(user: str = None):
        """Return the UID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            Any: The operation result.
        """
        return 0


    def get_gid(user: str = None):
        """Return the GID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            Any: The operation result.
        """
        return 0


    def get_shell_path():
        """Return the shell path for the current user.

        Returns:
            Any: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        import shutil
        shell_path = shutil.which("powershell") or shutil.which("cmd")
        if shell_path:
            return shell_path
        if "ComSpec" in os.environ:
            shell_path = os.environ["ComSpec"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        raise NotImplementedError(f"Unsupported system `{get_system()}`")

elif is_unix_like():

    def get_user(uid: int = None) -> str:
        """Return the user name for a UID or the current user.

        Args:
            uid (int): User id to resolve. If omitted, the current user is used.

        Returns:
            str: The operation result.
        """
        if uid is not None:
            import pwd
            return pwd.getpwuid(int(uid)).pw_name
        import getpass
        return getpass.getuser()


    def get_uid(user: str = None):
        """Return the UID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            Any: The operation result.
        """
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_uid
        return os.getuid()


    def get_gid(user: str = None):
        """Return the GID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            Any: The operation result.
        """
        if user is not None:
            import pwd
            return pwd.getpwnam(str(user)).pw_gid
        return os.getgid()


    def get_shell_path():
        """Return the shell path for the current user.

        Returns:
            Any: The operation result.
        """
        if "SHELL" in os.environ:
            shell_path = os.environ["SHELL"]
            if shell_path and os.path.exists(shell_path):
                return shell_path
        try:
            import pwd
            return pwd.getpwnam(get_user()).pw_shell
        except:
            import shutil
            return shutil.which("zsh") or shutil.which("bash") or shutil.which("sh")

else:

    def get_user(uid: int = None) -> str:
        """Return the user name for a UID or the current user.

        Args:
            uid (int): User id to resolve. If omitted, the current user is used.

        Returns:
            str: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        raise NotImplementedError(f"Unsupported system `{get_system()}`")


    def get_uid(user: str = None) -> int:
        """Return the UID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            int: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        raise NotImplementedError(f"Unsupported system `{get_system()}`")


    def get_gid(user: str = None) -> int:
        """Return the GID for a user or the current user.

        Args:
            user (str): User name to resolve. If omitted, the current user is used.

        Returns:
            int: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        raise NotImplementedError(f"Unsupported system `{get_system()}`")


    def get_shell_path() -> str:
        """Return the shell path for the current user.

        Returns:
            str: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        raise NotImplementedError(f"Unsupported system `{get_system()}`")


def import_module(name: str, spec: "ModuleSpec" = None) -> "T":
    """Import a module, optionally using an import spec.

    Args:
        name (str): Name to resolve.
        spec (ModuleSpec): The spec value.

    Returns:
        T: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    from importlib.util import find_spec, LazyLoader, module_from_spec
    if name in sys.modules:
        return sys.modules[name]
    spec = spec or find_spec(name)
    if not spec:
        raise ModuleNotFoundError(f"No module named '{name}'")
    loader = LazyLoader(spec.loader)
    spec.loader = loader
    module = module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def import_module_file(name: str, path: str) -> "T":
    """Import a module from a file path.

    Args:
        name (str): Name to resolve.
        path (str): Filesystem path to process.

    Returns:
        T: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    from importlib.util import LazyLoader, module_from_spec, spec_from_file_location
    if name in sys.modules:
        return sys.modules[name]
    if os.path.isdir(path):
        path = os.path.join(path, "__init__.py")
    if not os.path.exists(path):
        raise ModuleNotFoundError(f"No such file or directory: '{path}'")
    spec = spec_from_file_location(name, path)
    if not spec:
        raise ModuleNotFoundError(f"No module named '{name}'")
    loader = LazyLoader(spec.loader)
    spec.loader = loader
    module = module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def get_derived_type(t: "type[T]") -> "type[T]":
    """Create a proxy type that delegates operations to another type.

    Args:
        t (Type[T]): The t value.

    Returns:
        Type[T]: The operation result.
    """

    class Derived(Proxy):

        def __init__(self, obj: "T"):
            super().__init__()
            object.__setattr__(self, "__super__", obj)

        def _get_current_object(self):
            return self.__super__

    return Derived


def lazy_load(fn: "Callable[P, T]", *args: "P.args", **kwargs: "P.kwargs") -> "T":
    """Return a proxy that loads its target lazily.

    Args:
        fn (Callable[P, T]): Callable to invoke.
        args (P.args): Arguments passed to the operation.
        kwargs (P.kwargs): Keyword arguments passed to the operation.

    Returns:
        T: The operation result.
    """
    return Proxy(functools.partial(fn, *args, **kwargs))


def raise_error(e: "BaseException"):
    """Raise the provided exception instance.

    Args:
        e (BaseException): The e value.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    raise e


def lazy_raise(e: "BaseException") -> "T":
    """Return a proxy that raises the supplied exception when accessed.

    Args:
        e (BaseException): The e value.

    Returns:
        T: The operation result.
    """
    return lazy_load(raise_error, e)


@timeoutable
def wait_event(event: "threading.Event", timeout: "TimeoutType") -> bool:
    """Wait for a threading event with timeout handling.

    Args:
        event (threading.Event): Event name to register or trigger.
        timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

    Returns:
        bool: The operation result.
    """
    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return False
        if event.wait(min(t, interval)):
            return True


@timeoutable
def wait_thread(thread: "threading.Thread", timeout: "TimeoutType") -> bool:
    """Wait for a thread to finish with timeout handling.

    Args:
        thread (threading.Thread): The thread value.
        timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

    Returns:
        bool: The operation result.
    """
    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return False
        try:
            thread.join(min(t, interval))
        except:
            pass
        if not thread.is_alive():
            return True


@timeoutable
def wait_process(process: "subprocess.Popen", timeout: "TimeoutType") -> "int | None":
    """Wait for a process to finish with timeout handling.

    Args:
        process (subprocess.Popen): The process value.
        timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

    Returns:
        Optional[int]: The operation result.
    """
    import subprocess
    interval = 1
    while True:
        t = timeout.remain
        if t is None:
            t = interval
        elif t <= 0:
            return None
        try:
            return process.wait(min(t, interval))
        except subprocess.TimeoutExpired:
            pass


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
