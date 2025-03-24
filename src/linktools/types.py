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

if _t.TYPE_CHECKING:
    P = _t.ParamSpec("P")

PathType = _t.Union[str, _Path]
QueryDataType = _t.Union[str, int, float]
QueryType = _t.Union[QueryDataType, _t.List[QueryDataType], _t.Tuple[QueryDataType]]
TimeoutType = _t.Union["Timeout", float, int, None]

_logger: "_t.Optional[_logging.Logger]" = None


def _get_logger() -> "_logging.Logger":
    global _logger
    if _logger is None:
        from ._environ import environ
        _logger = environ.get_logger()
    return _logger


class Error(Exception):
    pass


class DownloadError(Error):
    pass


class DownloadHttpError(DownloadError):

    def __init__(self, code, e):
        super().__init__(e)
        self.code = code


class ConfigError(Error):
    pass


class ToolError(Error):
    pass


class ToolNotFound(ToolError):
    pass


class ToolNotSupport(ToolError):
    pass


class ToolExecError(ToolError):
    pass


class NoFreePortFoundError(Error):
    """Exception indicating that no free port could be found."""


def get_origin(tp):
    if hasattr(_t, "get_origin"):
        return _t.get_origin(tp)
    if tp is _t.Generic:
        return _t.Generic
    if isinstance(tp, _types.UnionType):
        return _types.UnionType
    if hasattr(tp, "__origin__"):
        return tp.__origin__
    raise TypeError(f"{tp} has no attribute '__origin__'")


def get_args(tp):
    if hasattr(_t, "get_args"):
        return _t.get_args(tp)
    if hasattr(tp, "__args__"):
        return tp.__args__
    raise TypeError(f"{tp} has no attribute '__args__'")


class Timeout:

    def __new__(cls, timeout: TimeoutType = None):
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
    def remain(self) -> _t.Union[float, None]:
        timeout = None
        if self._deadline is not None:
            timeout = max(self._deadline - _time.time(), 0)
        return timeout

    @property
    def deadline(self) -> _t.Union[float, None]:
        return self._deadline

    def reset(self) -> None:
        if self._timeout is not None and self._timeout >= 0:
            self._deadline = _time.time() + self._timeout

    def check(self) -> bool:
        if self._deadline is not None:
            if _time.time() > self._deadline:
                return False
        return True

    def ensure(self, err_type: "_t.Callable[[str], Exception]" = TimeoutError, message: str = "Timeout") -> None:
        if not self.check():
            raise err_type(message)

    def __repr__(self):
        return f"Timeout(timeout={self._timeout})"


class Stoppable(_abc.ABC):
    """
    Stoppable interface
    """

    @_abc.abstractmethod
    def stop(self):
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
    def lock(self) -> _threading.RLock():
        return self._lock


_event_handler_lock = _threading.RLock()
_event_handler_name = "__EventHandlerMixin_event_handler"


class EventHandlerMixin(object):

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

    def trigger(self, event: str, *args: _t.Any, **kwargs: _t.Any):
        logger = _get_logger()
        handler = self._event_handler
        invoke_list, remove_list = [], []
        with handler.lock:
            if event in handler:
                callbacks = handler.get(event)
                for callback, info in callbacks.items():
                    invoke_list.append(callback)
                    info["time"] += 1
                    if info["max_times"] is not None and info["times"] >= info["max_times"]:
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
    """
    A thread-safe, generic data queue for producer-consumer patterns.
    """

    def __init__(self, size: int):
        self._lock = _threading.RLock()  # Recursive lock for thread-safe operations
        self._size = size
        self._queue = _collections.deque([])
        self._last_put_time = 0  # Timestamp of the last put operation
        self._last_get_time = 0  # Timestamp of the last get operation

    def put(self, item: T) -> _t.Optional[T]:
        """
        Store an item in the queue and update the put timestamp.
        """
        with self._lock:
            result = None
            if 0 <= self._size <= len(self._queue):
                result = self._queue.popleft()
            self._queue.append(item)
            self._last_put_time = int(_time.time())
            return result

    def get(self) -> _t.Optional[T]:
        """
        Retrieve the item from the queue if available and update the get timestamp.
        """
        with self._lock:
            if len(self._queue) > 0:
                self._last_get_time = int(_time.time())
                return self._queue.popleft()
            return None

    def peek(self) -> _t.Optional[T]:
        """
        View the current item in the queue without updating the get timestamp.
        """
        with self._lock:
            if len(self._queue) > 0:
                return self._queue[0]
            return None

    def is_backlogged(self, timeout: int) -> bool:
        """
        Check if the item in the queue has been waiting for more than the given timeout.
        """
        with self._lock:
            if len(self._queue) == 0:
                return False
            return self._last_get_time + timeout < int(_time.time())

    def is_starving(self, timeout: int) -> bool:
        """
        Check if the queue has not received new items for more than the given timeout.
        """
        with self._lock:
            return self._last_put_time + timeout < int(_time.time())

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        """
        with self._lock:
            return len(self._queue) == 0

    def clear(self) -> None:
        """
        Clear the queue.
        """
        with self._lock:
            self._queue.clear()
            self._last_put_time = 0
            self._last_get_time = 0


_basic_cache_types = (type(None), int, float, bool, complex)


def _filter_cache_items(d: _t.Dict[_t.Any, _t.Any]) -> _t.Dict[_t.Any, _t.Any]:
    return {k: v for k, v in d.items() if v}


class _FileCacheLock:

    def __init__(self, cache: "FileCache", type: str, key: str):
        from filelock import FileLock
        path = cache.directory / type
        path.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(path / f"{key or ''}.lock"))

    def acquire(self, timeout: TimeoutType = None):
        self._lock.acquire(Timeout(timeout).remain)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


class _FileCacheData(_t.Generic[T]):

    def __init__(self, cache: "FileCache", type: str):
        self._cache = cache
        self._lock = _FileCacheLock(cache, type, "data")
        self._lock.acquire()
        self._data = _shelve.open(str(cache.directory / type / "data"))

    def close(self) -> None:
        self._data.close()
        self._lock.release()

    def set(self, key: str, value: T, ttl: int = None) -> None:
        self._data[key] = _filter_cache_items({
            "data": self._cache.dump(value),
            "ttl": int(ttl) if ttl else None,
            "ts": int(_time.time()),
        })

    def update(self, **kwargs: T) -> None:
        for key, value in kwargs.items():
            self.set(key, value)

    def _get(self, key: str) -> _t.Optional[_t.Dict[str, _t.Any]]:
        value = self._data.get(key, None)
        if value:
            timestamp = value.get("ts", None)
            ttl = value.get("ttl", None)
            if timestamp and ttl and timestamp + ttl < _time.time():
                self._data.pop(key)
                return None
            return value
        return None

    def get(self, key: str, default: _t.Any = None) -> _t.Optional[T]:
        value = self._get(key)
        if value:
            return self._cache.load(value.get("data", None))
        return default

    def pop(self, key: str, default: _t.Any = None) -> _t.Optional[T]:
        value = self._get(key)
        if value:
            self._data.pop(key)
            return self._cache.load(value.get("data", None))
        return default

    def peek(self) -> _t.Optional[str]:
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                return key
        return None

    def peekitem(self) -> _t.Tuple[_t.Optional[str], _t.Optional[T]]:
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                return key, self._cache.load(value.get("data", None))
        return None, None

    def incr(self, key: str, delta: int = 1, default: int = 0) -> int:
        value = self._get(key)
        if value:
            result = self._cache.load(value.get("data", None))
            if not isinstance(result, (int, float)):
                raise TypeError(f"the value of key `{key}` is not int")
            result = result + delta
            value["data"] = self._cache.dump(result)
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

    def items(self) -> "_t.Generator[_t.Tuple[str, T], None, None]":
        for key in list(self._data.keys()):
            value = self._get(key)
            if value:
                yield key, self._cache.load(value.get("data", None))

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

    def __init__(self, directory: PathType, load: _t.Callable[[str], T] = None, dump: _t.Callable[[T], str] = None):
        self._directory = _Path(directory)
        self._load = load
        self._dump = dump

    @property
    def directory(self) -> _Path:
        return self._directory

    def lock(self, key: str) -> "_FileCacheLock":
        return _FileCacheLock(self, "lock", key)

    def open(self) -> "_FileCacheData[T]":
        return _FileCacheData(self, "data")

    def set(self, key: str, value: T, ttl: int = None) -> None:
        with self.open() as data:
            data.set(key, value, ttl)

    def update(self, **kwargs: T) -> None:
        with self.open() as data:
            data.update(**kwargs)

    def get(self, key: str, default: _t.Any = None) -> _t.Optional[T]:
        with self.open() as data:
            return data.get(key, default=default)

    def pop(self, key: str, default: _t.Any = None) -> _t.Optional[T]:
        with self.open() as data:
            return data.pop(key, default=default)

    def peek(self) -> _t.Optional[T]:
        with self.open() as data:
            return data.peek()

    def peekitem(self) -> _t.Tuple[_t.Optional[str], _t.Optional[T]]:
        with self.open() as data:
            return data.peekitem()

    def incr(self, key: str, delta: int = 1, default: int = 0) -> int:
        with self.open() as data:
            return data.incr(key, delta, default)

    def keys(self) -> "_t.Generator[str, None, None]":
        with self.open() as data:
            yield from data.keys()

    def items(self) -> "_t.Generator[_t.Tuple[str, T], None, None]":
        with self.open() as data:
            yield from data.items()

    def load(self, data: _t.Any) -> T:
        if self._load:
            if not isinstance(data, _basic_cache_types):
                return self._load(data)
        return data

    def dump(self, data: T) -> _t.Any:
        if self._dump:
            if not isinstance(data, _basic_cache_types):
                return self._dump(data)
        return data

    def __len__(self) -> int:
        with self.open() as data:
            return len(data)


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
