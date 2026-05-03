#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : types.py
@time    : 2024/7/21
@site    : https://github.com/ice-black-tea
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
import abc as _abc
import collections as _collections
import logging as _logging
import shelve as _shelve
import threading as _threading
import time as _time
import types as _types
import typing as _t
import weakref as _weakref
from pathlib import Path as _Path

T = _t.TypeVar("T")
PathType = _t.Union[str, _Path]
QueryDataType = _t.Union[str, int, float]
QueryType = _t.Union[QueryDataType, _t.List[QueryDataType], _t.Tuple[QueryDataType]]
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


class Error(Exception):
    """Base exception for linktools-specific errors."""
    pass


class ModuleError(Error):
    """Raised when a linktools module cannot be loaded or used."""
    pass


class DownloadError(Error):
    """Base exception for download failures."""
    pass


class ExecError(Error):
    """Base exception for process execution failures."""
    pass


class DownloadHttpError(DownloadError):

    """Download error that carries an HTTP status code."""
    def __init__(self, code, e):
        super().__init__(e)
        self.code = code


class ConfigError(Error):
    """Raised when configuration data is invalid or unavailable."""
    pass


class ToolError(Error):
    """Base exception for tool discovery and execution failures."""
    pass


class ToolNotFound(ToolError):
    """Raised when a requested tool cannot be found."""
    pass


class ToolNotSupport(ToolError):
    """Raised when a tool is not supported in the current environment."""
    pass


class ToolExecError(ToolError):
    """Raised when a tool process exits with an execution error."""
    pass


class NoFreePortFoundError(Error):
    """Exception indicating that no free port could be found."""


def get_origin(tp):
    """Return the unsubscripted origin for a typing object.

    Args:
        tp: The tp value.

    Returns:
        Any: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
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
    """Return the type arguments for a typing object.

    Args:
        tp: The tp value.

    Returns:
        Any: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
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
        elif isinstance(timeout, (float, int, type(None))):
            t = super().__new__(cls)
            t._timeout = timeout
            t._deadline = None
            t.reset()
            return t
        raise TypeError(f"Timeout/int/float was expects, got {type(timeout)}")

    @property
    def remain(self) -> "float | None":
        """Remain.

        Returns:
            _t.Union[float, None]: The property value.
        """
        timeout = None
        if self._deadline is not None:
            timeout = max(self._deadline - _time.time(), 0)
        return timeout

    @property
    def deadline(self) -> "float | None":
        """Deadline.

        Returns:
            _t.Union[float, None]: The property value.
        """
        return self._deadline

    def reset(self) -> None:
        """Reset the deadline from the configured timeout."""
        if self._timeout is not None and self._timeout >= 0:
            self._deadline = _time.time() + self._timeout

    def check(self) -> bool:
        """Return whether the timeout has not expired.

        Returns:
            bool: The operation result.
        """
        if self._deadline is not None:
            if _time.time() > self._deadline:
                return False
        return True

    def ensure(self, err_type: "_t.Callable[[str], Exception]" = TimeoutError, message: str = "Timeout") -> None:
        """Raise an error if the timeout has expired.

        Args:
            err_type (_t.Callable[[str], Exception]): The err_type value.
            message (str): The message value.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
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
        """Register an event callback.

        Args:
            event (str): Event name to register or trigger.
            callback (_t.Callable[..., _t.Any]): Callback invoked for the event.
            times (int): The times value.
        """
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
        """Unregister an event callback.

        Args:
            event (str): Event name to register or trigger.
            callback (_t.Callable[..., _t.Any]): Callback invoked for the event.
        """
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
        """Register an event callback that runs once.

        Args:
            event (str): Event name to register or trigger.
            callback (_t.Callable[..., _t.Any]): Callback invoked for the event.
        """
        self.on(event, callback, 1)

    def trigger(self, event: str, *args: "_t.Any", **kwargs: "_t.Any"):
        """Trigger an event and invoke registered callbacks.

        Args:
            event (str): Event name to register or trigger.
            args (_t.Any): Arguments passed to the operation.
            kwargs (_t.Any): Keyword arguments passed to the operation.
        """
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
            del remove_list
        logger.debug(f"Event `{event}` invoke {len(invoke_list)} callbacks")
        for callback in invoke_list:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Event `{event}` handler `{callback}` error", exc_info=e)


class SlidingQueue(_t.Generic[T]):
    """A thread-safe, generic data queue for producer-consumer patterns."""

    def __init__(self, size: int):
        self._lock = _threading.RLock()  # Recursive lock for thread-safe operations
        self._size = size
        self._queue = _collections.deque([])
        self._last_put_time = 0  # Timestamp of the last put operation
        self._last_get_time = 0  # Timestamp of the last get operation

    def put(self, item: "T") -> "T | None":
        """Store an item in the queue and update the put timestamp.

        Args:
            item (T): The item value.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self._lock:
            result = None
            if 0 <= self._size <= len(self._queue):
                result = self._queue.popleft()
            self._queue.append(item)
            self._last_put_time = int(_time.time())
            return result

    def get(self) -> "T | None":
        """Retrieve the item from the queue if available and update the get timestamp.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self._lock:
            if len(self._queue) > 0:
                self._last_get_time = int(_time.time())
                return self._queue.popleft()
            return None

    def peek(self) -> "T | None":
        """View the current item in the queue without updating the get timestamp.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self._lock:
            if len(self._queue) > 0:
                return self._queue[0]
            return None

    def is_backlogged(self, timeout: int) -> bool:
        """Check if the item in the queue has been waiting for more than the given timeout.

        Args:
            timeout (int): Maximum time to wait, or None to wait indefinitely.

        Returns:
            bool: The operation result.
        """
        with self._lock:
            if len(self._queue) == 0:
                return False
            return self._last_get_time + timeout < int(_time.time())

    def is_starving(self, timeout: int) -> bool:
        """Check if the queue has not received new items for more than the given timeout.

        Args:
            timeout (int): Maximum time to wait, or None to wait indefinitely.

        Returns:
            bool: The operation result.
        """
        with self._lock:
            return self._last_put_time + timeout < int(_time.time())

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: The operation result.
        """
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self._queue.clear()
            self._last_put_time = 0
            self._last_get_time = 0


_basic_cache_types = (type(None), int, float, bool, complex)


def _filter_cache_items(d: "dict[_t.Any, _t.Any]") -> "dict[_t.Any, _t.Any]":
    return {k: v for k, v in d.items() if v}


class _FileCacheLock:

    def __init__(self, cache: "FileCache", namespace: str, key: str = None):
        from filelock import FileLock
        path = cache.directory / namespace
        path.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(path / f"{key or ''}.lock"))

    def acquire(self, timeout: "TimeoutType" = None):
        self._lock.acquire(Timeout(timeout).remain)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


_file_cache_backup_suffix = ".backup"

class _FileCacheBackup:

    def __init__(self, cache: "FileCache", namespace: str):
        self._directory = cache.directory / namespace
        self._lock = _FileCacheLock(cache, namespace, "backup")
        self._lock.acquire()

    def close(self) -> None:
        self._lock.release()

    def backup(self, path: "PathType", version: str = None, max_count: int = 3):
        import os
        import shutil
        from datetime import datetime
        from . import utils

        if not version:
            version = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{utils.make_uuid()[:12]}"

        self._directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, self._get_version_path(version))

        versions = self.list_versions()
        if len(versions) > max(max_count, 1):
            os.remove(self._get_version_path(versions[0]))

        return version

    def restore(self, path: "PathType", version: str = None):
        import os
        import shutil

        if not version:
            versions = self.list_versions()
            if not versions:
                raise Exception("Not found any backup version")
            version = versions[-1]

        if not os.path.isfile(self._get_version_path(version)):
            raise Exception(f"Not found backup version `{version}`")

        shutil.copy2(self._get_version_path(version), path)

        return version

    def list_versions(self) -> "list[str]":
        import os

        if not self._directory.is_dir():
            return []

        versions = {}
        for name in os.listdir(self._directory):
            version = self._parse_version_name(name)
            if version:
                versions[version] = os.path.getctime(self._directory / name)

        return sorted(versions.keys(), key=lambda o: versions[o])

    def _get_version_path(self, version: str) -> "_Path":
        return self._directory / f"{version}{_file_cache_backup_suffix}"

    @classmethod
    def _parse_version_name(cls, name: str):
        if name.endswith(_file_cache_backup_suffix):
            return name[:-len(_file_cache_backup_suffix)]
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _FileCacheData(_t.Generic[T]):

    def __init__(self, cache: "FileCache", namespace: str):
        self._cache = cache
        self._lock = _FileCacheLock(cache, namespace, "data")
        self._lock.acquire()
        self._data = _shelve.open(str(cache.directory / namespace / "data"))

    def close(self) -> None:
        self._data.close()
        self._lock.release()

    def set(self, key: str, value: "T", ttl: int = None) -> None:
        self._data[key] = _filter_cache_items({
            "data": self._cache.serialize(value),
            "ttl": int(ttl) if ttl else None,
            "ts": int(_time.time()),
        })

    def update(self, **kwargs: "T") -> None:
        for key, value in kwargs.items():
            self.set(key, value)

    def _get(self, key: str) -> "dict[str, _t.Any] | None":
        value = self._data.get(key, None)
        if value:
            timestamp = value.get("ts", None)
            ttl = value.get("ttl", None)
            if timestamp and ttl and timestamp + ttl < _time.time():
                self._data.pop(key)
                return None
            return value
        return None

    def get(self, key: str, default: "_t.Any" = None) -> "T | None":
        value = self._get(key)
        if value:
            return self._cache.unserialize(value.get("data", None))
        return default

    def pop(self, key: str, default: "_t.Any" = None) -> "T | None":
        value = self._get(key)
        if value:
            self._data.pop(key)
            return self._cache.unserialize(value.get("data", None))
        return default

    def peek(self) -> "str | None":
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                return key
        return None

    def peekitem(self) -> "tuple[str | None, T | None]":
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                return key, self._cache.unserialize(value.get("data", None))
        return None, None

    def incr(self, key: str, delta: int = 1, default: int = 0) -> int:
        value = self._get(key)
        if value:
            result = self._cache.unserialize(value.get("data", None))
            if not isinstance(result, (int, float)):
                raise TypeError(f"the value of key `{key}` is not int")
            result = result + delta
            value["data"] = self._cache.serialize(result)
            value["ts"] = int(_time.time())
            self._data[key] = value
        else:
            result = default
            self.set(key, result)
        return result

    def keys(self) -> "_t.Generator[str, None, None]":
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                yield key

    def items(self) -> "_t.Generator[tuple[str, T], None, None]":
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                yield key, self._cache.unserialize(value.get("data", None))

    def __len__(self) -> int:
        count = 0
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                count += 1
        return count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FileCache(_t.Generic[T]):

    """Persist and retrieve cached values in a local file."""
    def __init__(self, directory: "PathType", *,
                 serialize: "_t.Callable[[str], T]" = None, unserialize: "_t.Callable[[T], str]" = None,
                 dump: "_t.Callable[[T], str]" = None, load: "_t.Callable[[str], T]" = None):
        if dump or load:
            _get_logger().warning("deprecated dump and load, use serialize and unserialize instead")
        self._directory = _Path(directory)
        self._serialize = serialize or dump
        self._unserialize = unserialize or load

    @property
    def directory(self) -> "_Path":
        """Directory.

        Returns:
            _Path: The property value.
        """
        return self._directory

    def lock(self, key: str = None) -> "_FileCacheLock":
        """Open a lock for cache operations.

        Args:
            key (str): Configuration or item key.

        Returns:
            _FileCacheLock: The operation result.
        """
        return _FileCacheLock(self, "lock", key)

    def backup(self) -> "_FileCacheBackup":
        """Open a backup context for cache files.

        Returns:
            _FileCacheBackup: The operation result.
        """
        return _FileCacheBackup(self, "backup")

    def open(self) -> "_FileCacheData[T]":
        """Open the cache data context.

        Returns:
            _FileCacheData[T]: The operation result.
        """
        return _FileCacheData(self, "data")

    def set(self, key: str, value: "T", ttl: int = None) -> None:
        """Store a cache value by key.

        Args:
            key (str): Configuration or item key.
            value (T): Value to store or process.
            ttl (int): The ttl value.
        """
        with self.open() as data:
            data.set(key, value, ttl)

    def update(self, **kwargs: "T") -> None:
        """Update multiple cache values.

        Args:
            kwargs (T): Keyword arguments passed to the operation.
        """
        with self.open() as data:
            data.update(**kwargs)

    def get(self, key: str, default: "_t.Any" = None) -> "T | None":
        """Return a cache value by key.

        Args:
            key (str): Configuration or item key.
            default (_t.Any): Value returned when no explicit value is available.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self.open() as data:
            return data.get(key, default=default)

    def pop(self, key: str, default: "_t.Any" = None) -> "T | None":
        """Remove and return a cache value by key.

        Args:
            key (str): Configuration or item key.
            default (_t.Any): Value returned when no explicit value is available.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self.open() as data:
            return data.pop(key, default=default)

    def peek(self) -> "T | None":
        """Return the next cache value without removing it.

        Returns:
            _t.Optional[T]: The operation result.
        """
        with self.open() as data:
            return data.peek()

    def peekitem(self) -> "tuple[str | None, T | None]":
        """Return the next cache item without removing it.

        Returns:
            _t.Tuple[_t.Optional[str], _t.Optional[T]]: The operation result.
        """
        with self.open() as data:
            return data.peekitem()

    def incr(self, key: str, delta: int = 1, default: int = 0) -> int:
        """Increment a numeric cache value.

        Args:
            key (str): Configuration or item key.
            delta (int): The delta value.
            default (int): Value returned when no explicit value is available.

        Returns:
            int: The operation result.
        """
        with self.open() as data:
            return data.incr(key, delta, default)

    def keys(self) -> "_t.Generator[str, None, None]":
        """Yield cache keys.

        Returns:
            _t.Generator[str, None, None]: The operation result.
        """
        with self.open() as data:
            yield from data.keys()

    def items(self) -> "_t.Generator[tuple[str, T], None, None]":
        """Yield cache items.

        Returns:
            _t.Generator[_t.Tuple[str, T], None, None]: The operation result.
        """
        with self.open() as data:
            yield from data.items()

    def serialize(self, data: "T") -> "_t.Any":
        """Serialize a cache value for storage.

        Args:
            data (T): The data value.

        Returns:
            _t.Any: The operation result.
        """
        if self._serialize:
            if not isinstance(data, _basic_cache_types):
                return self._serialize(data)
        return data

    def unserialize(self, data: "_t.Any") -> "T":
        """Restore a cache value after loading.

        Args:
            data (_t.Any): The data value.

        Returns:
            T: The operation result.
        """
        if self._unserialize:
            if not isinstance(data, _basic_cache_types):
                return self._unserialize(data)
        return data

    def __len__(self) -> int:
        with self.open() as data:
            return len(data)


class _ReactorEvent:

    def __init__(self, fn: "_t.Callable[[], any]", when: float, interval: float):
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

    """Run submitted callables on a dedicated worker thread."""
    def __init__(self, on_stop=None, on_error=None):
        self._running = False
        self._on_stop = on_stop
        self._on_error = on_error
        self._lock = _threading.Lock()
        self._cond = _threading.Condition(self._lock)
        self._worker = None
        self._pending: "_collections.deque[_ReactorEvent]" = _collections.deque([])

    def is_running(self) -> bool:
        """Return whether the reactor worker is running.

        Returns:
            bool: The operation result.
        """
        with self._lock:
            return self._running

    def start(self):
        """Start the reactor worker thread."""
        if self._running:
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = _threading.Thread(target=self._run)
            self._worker.daemon = True
            self._worker.start()

    def run(self, timeout: "TimeoutType"):
        """Run the reactor until it stops or times out.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.
        """
        with self:
            self.wait(Timeout(timeout))

    def _run(self):
        running = True
        while running:
            now = _time.time()
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
                except BaseException as e:
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

    def stop(self):
        """Signal the reactor to stop and wait for it."""
        self.signal_stop()
        self.wait()

    def _stop(self):
        with self._lock:
            self._running = False

    def signal_stop(self, delay: float = None):
        """Schedule the reactor to stop.

        Args:
            delay (float): Delay before the operation runs.
        """
        self.schedule(self._stop, delay)

    def schedule(self, fn: "_t.Callable[[], any]", delay: float = None, interval: float = None):
        """Schedule a callable to run later or repeatedly.

        Args:
            fn (_t.Callable[[], any]): Callable to invoke.
            delay (float): Delay before the operation runs.
            interval (float): Interval used for repeated execution.
        """
        now = _time.time()
        if delay is not None:
            when = now + delay
        else:
            when = now
        with self._lock:
            item = _ReactorEvent(fn, when, interval)
            self._pending.append(item)
            self._cond.notify()

    def _work(self, fn: "_t.Callable[[], any]"):
        fn()

    def wait(self, timeout: "TimeoutType" = None) -> bool:
        """Wait for the reactor worker to finish.

        Args:
            timeout (TimeoutType): Maximum time to wait, or None to wait indefinitely.

        Returns:
            bool: The operation result.
        """
        from . import utils
        worker = self._worker
        if worker:
            if _threading.current_thread().ident == worker.ident:
                _logger.warning("Cannot wait on the reactor from its own thread")
                return False

            return utils.wait_thread(worker, timeout)
        return True

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Code stolen from celery.local.Proxy: https://github.com/celery/celery/blob/main/celery/local.py

def _default_cls_attr(name, type_, cls_value):
    # Proxy uses properties to forward the standard
    # class attributes __module__, __name__ and __doc__ to the real
    # object, but these needs to be a string when accessed from
    # the Proxy class directly.  This is a hack to make that work.
    # -- See Issue #1087.

    def __new__(cls, getter):
        instance = type_.__new__(cls, cls_value)
        instance.__getter = getter
        return instance

    def __get__(self, obj, cls=None):
        return self.__getter(obj) if obj is not None else self

    return type(name, (type_,), {
        '__new__': __new__, '__get__': __get__,
    })


__module__ = __name__  # used by Proxy class body

_proxy_fn = "_Proxy__fn"
_proxy_object = "_Proxy__object"


class Proxy(object):
    """Proxy to another object."""

    __slots__ = ('__fn', '__object', '__dict__')
    __missing__ = object()

    def __init__(self, fn=__missing__, name=None, doc=None):
        object.__setattr__(self, _proxy_fn, fn)
        object.__setattr__(self, _proxy_object, Proxy.__missing__)
        if name is not None:
            object.__setattr__(self, "__custom_name__", name)
        if doc is not None:
            object.__setattr__(self, "__doc__", doc)

    @_default_cls_attr('name', str, __name__)
    def __name__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__name__

    @_default_cls_attr('qualname', str, __name__)
    def __qualname__(self):
        try:
            return self.__custom_name__
        except AttributeError:
            return self._get_current_object().__qualname__

    @_default_cls_attr('module', str, __module__)
    def __module__(self):
        return self._get_current_object().__module__

    @_default_cls_attr('doc', str, __doc__)
    def __doc__(self):
        return self._get_current_object().__doc__

    def _get_class(self):
        return self._get_current_object().__class__

    @property
    def __class__(self):
        return self._get_class()

    def _get_current_object(self):
        obj = getattr(self, _proxy_object)
        if obj == Proxy.__missing__:
            obj = getattr(self, _proxy_fn)()
            object.__setattr__(self, _proxy_object, obj)
        return obj

    @property
    def __dict__(self):
        return self._get_current_object().__dict__

    def __repr__(self):
        return repr(self._get_current_object())

    def __bool__(self):
        return bool(self._get_current_object())

    __nonzero__ = __bool__  # Py2

    def __dir__(self):
        return dir(self._get_current_object())

    def __getattr__(self, name):
        if name == '__members__':
            return dir(self._get_current_object())
        return getattr(self._get_current_object(), name)

    def __setitem__(self, key, value):
        self._get_current_object()[key] = value

    def __delitem__(self, key):
        del self._get_current_object()[key]

    def __setslice__(self, i, j, seq):
        self._get_current_object()[i:j] = seq

    def __delslice__(self, i, j):
        del self._get_current_object()[i:j]

    def __setattr__(self, name, value):
        setattr(self._get_current_object(), name, value)

    def __delattr__(self, name):
        delattr(self._get_current_object(), name)

    def __str__(self):
        return str(self._get_current_object())

    def __lt__(self, other):
        return self._get_current_object() < other

    def __le__(self, other):
        return self._get_current_object() <= other

    def __eq__(self, other):
        return self._get_current_object() == other

    def __ne__(self, other):
        return self._get_current_object() != other

    def __gt__(self, other):
        return self._get_current_object() > other

    def __ge__(self, other):
        return self._get_current_object() >= other

    def __hash__(self):
        return hash(self._get_current_object())

    def __call__(self, *a, **kw):
        return self._get_current_object()(*a, **kw)

    def __len__(self):
        return len(self._get_current_object())

    def __getitem__(self, i):
        return self._get_current_object()[i]

    def __iter__(self):
        return iter(self._get_current_object())

    def __contains__(self, i):
        return i in self._get_current_object()

    def __getslice__(self, i, j):
        return self._get_current_object()[i:j]

    def __add__(self, other):
        return self._get_current_object() + other

    def __sub__(self, other):
        return self._get_current_object() - other

    def __mul__(self, other):
        return self._get_current_object() * other

    def __floordiv__(self, other):
        return self._get_current_object() // other

    def __mod__(self, other):
        return self._get_current_object() % other

    def __divmod__(self, other):
        return self._get_current_object().__divmod__(other)

    def __pow__(self, other):
        return self._get_current_object() ** other

    def __lshift__(self, other):
        return self._get_current_object() << other

    def __rshift__(self, other):
        return self._get_current_object() >> other

    def __and__(self, other):
        return self._get_current_object() & other

    def __xor__(self, other):
        return self._get_current_object() ^ other

    def __or__(self, other):
        return self._get_current_object() | other

    def __div__(self, other):
        return self._get_current_object().__div__(other)

    def __truediv__(self, other):
        return self._get_current_object().__truediv__(other)

    def __neg__(self):
        return -(self._get_current_object())

    def __pos__(self):
        return +(self._get_current_object())

    def __abs__(self):
        return abs(self._get_current_object())

    def __invert__(self):
        return ~(self._get_current_object())

    def __complex__(self):
        return complex(self._get_current_object())

    def __int__(self):
        return int(self._get_current_object())

    def __float__(self):
        return float(self._get_current_object())

    def __oct__(self):
        return oct(self._get_current_object())

    def __hex__(self):
        return hex(self._get_current_object())

    def __index__(self):
        return self._get_current_object().__index__()

    def __coerce__(self, other):
        return self._get_current_object().__coerce__(other)

    def __enter__(self):
        return self._get_current_object().__enter__()

    def __exit__(self, *a, **kw):
        return self._get_current_object().__exit__(*a, **kw)

    def __reduce__(self):
        return self._get_current_object().__reduce__()


class IterProxy(_t.Iterable):
    """Proxy iterable operations to a lazily resolved object."""
    __missing__ = object()

    def __init__(self, func: "_t.Callable[P, _t.Iterable[T]]", *args: "P.args", **kwargs: "P.kwargs"):
        self._data = IterProxy.__missing__
        self._fn = func
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        if self._data == IterProxy.__missing__:
            self._data = self._fn(*self._args, **self._kwargs)
        return iter(self._data)
