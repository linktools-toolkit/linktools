#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shelve as _shelve
import time as _time
import typing as _t
from pathlib import Path as _Path

from .types import PathType, T, Timeout, TimeoutType

_basic_cache_types = (type(None), int, float, bool, complex)
_file_cache_backup_suffix = ".backup"


def _filter_cache_items(d: "dict[_t.Any, _t.Any]") -> "dict[_t.Any, _t.Any]":
    return {k: v for k, v in d.items() if v}


class _CacheLock:

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


class _BackupManager:

    def __init__(self, cache: "FileCache", namespace: str):
        self._directory = cache.directory / namespace
        self._lock = _CacheLock(cache, namespace, "backup")
        self._lock.acquire()

    def close(self) -> None:
        self._lock.release()

    def create(self, path: "PathType", version: str = None, max_count: int = 3):
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

    def backup(self, path: "PathType", version: str = None, max_count: int = 3):
        return self.create(path, version=version, max_count=max_count)

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


class _CacheSession(_t.Generic[T]):

    def __init__(self, cache: "FileCache", namespace: str):
        self._cache = cache
        self._lock = _CacheLock(cache, namespace, "data")
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

    def delete(self, key: str) -> bool:
        value = self._get(key)
        if value:
            self._data.pop(key, None)
            return True
        return False

    def pop(self, key: str, default: "_t.Any" = None) -> "T | None":
        value = self._get(key)
        if value:
            self._data.pop(key)
            return self._cache.unserialize(value.get("data", None))
        return default

    def contains(self, key: str) -> bool:
        return self._get(key) is not None

    def clear(self) -> None:
        self._data.clear()

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

    def __init__(self, directory: "PathType", *,
                 serialize: "_t.Callable[[str], T]" = None, unserialize: "_t.Callable[[T], str]" = None):
        self._directory = _Path(directory)
        self._serialize = serialize
        self._unserialize = unserialize

    @property
    def directory(self) -> "_Path":
        return self._directory

    def lock(self, key: str = None) -> "_CacheLock":
        return _CacheLock(self, "lock", key)

    def backups(self) -> "_BackupManager":
        return _BackupManager(self, "backup")

    def session(self) -> "_CacheSession[T]":
        return _CacheSession(self, "data")

    def set(self, key: str, value: "T", ttl: int = None) -> None:
        with self.session() as data:
            data.set(key, value, ttl)

    def update(self, **kwargs: "T") -> None:
        with self.session() as data:
            data.update(**kwargs)

    def get(self, key: str, default: "_t.Any" = None) -> "T | None":
        with self.session() as data:
            return data.get(key, default=default)

    def delete(self, key: str) -> bool:
        with self.session() as data:
            return data.delete(key)

    def pop(self, key: str, default: "_t.Any" = None) -> "T | None":
        with self.session() as data:
            return data.pop(key, default=default)

    def contains(self, key: str) -> bool:
        with self.session() as data:
            return data.contains(key)

    def clear(self) -> None:
        with self.session() as data:
            data.clear()

    def peek(self) -> "T | None":
        with self.session() as data:
            return data.peek()

    def peekitem(self) -> "tuple[str | None, T | None]":
        with self.session() as data:
            return data.peekitem()

    def incr(self, key: str, delta: int = 1, default: int = 0) -> int:
        with self.session() as data:
            return data.incr(key, delta, default)

    def keys(self) -> "_t.Generator[str, None, None]":
        with self.session() as data:
            yield from data.keys()

    def items(self) -> "_t.Generator[tuple[str, T], None, None]":
        with self.session() as data:
            yield from data.items()

    def serialize(self, data: "T") -> "_t.Any":
        if self._serialize and not isinstance(data, _basic_cache_types):
            return self._serialize(data)
        return data

    def unserialize(self, data: "_t.Any") -> "T":
        if self._unserialize and not isinstance(data, _basic_cache_types):
            return self._unserialize(data)
        return data

    def __len__(self) -> int:
        with self.session() as data:
            return len(data)
