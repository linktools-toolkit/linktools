#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence backend and retention values independent of Runtime domains."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ._cache_contracts import CacheCapabilities

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


class StorageRetentionMode(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    DURABLE = "durable"
    VOLATILE = "volatile"
    TRANSIENT = "transient"


class _StorageBackendKind(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    SQL = "sql"


@dataclass(frozen=True, slots=True, init=False)
class StorageRoute:
    _kind: _StorageBackendKind
    _retention: StorageRetentionMode
    _path: "Path | None"
    _transaction_root: "Path | None"
    _engine: "AsyncEngine | None"
    _cache: "CacheCapabilities | None" = field(compare=False, hash=False, repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use StorageRoute factory methods")

    @classmethod
    def _create(
        cls,
        *,
        kind: _StorageBackendKind,
        retention: StorageRetentionMode,
        path: "Path | None" = None,
        transaction_root: "Path | None" = None,
        engine: "AsyncEngine | None" = None,
        cache: "CacheCapabilities | None" = None,
    ) -> "StorageRoute":
        value = object.__new__(cls)
        object.__setattr__(value, "_kind", kind)
        object.__setattr__(value, "_retention", retention)
        object.__setattr__(value, "_path", path)
        object.__setattr__(value, "_transaction_root", transaction_root)
        object.__setattr__(value, "_engine", engine)
        object.__setattr__(value, "_cache", cache)
        value._validate()
        return value

    def _validate(self) -> None:
        valid = (
            (
                self._kind is _StorageBackendKind.MEMORY
                and self._retention in {StorageRetentionMode.VOLATILE, StorageRetentionMode.TRANSIENT}
                and self._path is None
                and self._transaction_root is None
                and self._engine is None
            )
            or (
                self._kind is _StorageBackendKind.FILESYSTEM
                and self._retention is StorageRetentionMode.DURABLE
                and self._path is not None
                and self._engine is None
                and (
                    self._transaction_root is None
                    or isinstance(self._transaction_root, Path)
                )
            )
            or (
                self._kind is _StorageBackendKind.SQLITE
                and self._retention is StorageRetentionMode.DURABLE
                and self._path is not None
                and self._transaction_root is None
                and self._engine is None
            )
            or (
                self._kind is _StorageBackendKind.SQL
                and self._retention is StorageRetentionMode.DURABLE
                and self._path is None
                and self._transaction_root is None
                and self._engine is not None
            )
        )
        if not valid:
            raise ValueError("StorageRoute has an invalid backend and retention combination")

    @property
    def kind(self) -> str:
        return self._kind.value

    @property
    def retention(self) -> StorageRetentionMode:
        return self._retention

    @property
    def path(self) -> "Path | None":
        return self._path

    @property
    def transaction_root(self) -> "Path | None":
        return self._transaction_root

    @property
    def engine(self) -> "AsyncEngine | None":
        return self._engine

    @property
    def cache(self) -> "CacheCapabilities | None":
        """Borrowed optional cache, independent of persistence identity."""
        return self._cache

    def with_cache(self, cache: "CacheCapabilities | None") -> "StorageRoute":
        """Copy this persistence route while borrowing a different cache."""
        return type(self)._create(
            kind=self._kind, retention=self._retention, path=self._path,
            transaction_root=self._transaction_root, engine=self._engine, cache=cache,
        )

    @classmethod
    def memory(cls, *, cache: "CacheCapabilities | None" = None) -> "StorageRoute":
        return cls._create(kind=_StorageBackendKind.MEMORY, retention=StorageRetentionMode.VOLATILE, cache=cache)

    @classmethod
    def transient(cls, *, cache: "CacheCapabilities | None" = None) -> "StorageRoute":
        return cls._create(kind=_StorageBackendKind.MEMORY, retention=StorageRetentionMode.TRANSIENT, cache=cache)

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        transaction_root: "str | Path | None" = None,
        cache: "CacheCapabilities | None" = None,
    ) -> "StorageRoute":
        return cls._create(
            kind=_StorageBackendKind.FILESYSTEM,
            retention=StorageRetentionMode.DURABLE,
            path=_normalize_path(path),
            cache=cache,
            transaction_root=None if transaction_root is None else _normalize_path(transaction_root),
        )

    @classmethod
    def sqlite(cls, path: "str | Path", *, cache: "CacheCapabilities | None" = None) -> "StorageRoute":
        if not isinstance(path, (str, Path)) or not str(path).strip() or str(path).strip() == ":memory:":
            raise ValueError("StorageRoute.sqlite requires a filesystem path")
        normalized = _normalize_path(path)
        return cls._create(kind=_StorageBackendKind.SQLITE, retention=StorageRetentionMode.DURABLE, path=normalized, cache=cache)

    @classmethod
    def sql(cls, engine: "AsyncEngine", *, cache: "CacheCapabilities | None" = None) -> "StorageRoute":
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not isinstance(engine, AsyncEngine):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "SQL route requires an AsyncEngine")
        if engine.dialect.name == "sqlite" and engine.url.database in {None, "", ":memory:"}:
            raise ValueError("external SQL route requires durable SQLite or external SQL")
        return cls._create(kind=_StorageBackendKind.SQL, retention=StorageRetentionMode.DURABLE, engine=engine, cache=cache)

def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("StorageRoute path is required")
    return Path(value).expanduser().resolve(strict=False)


__all__ = ["StorageRoute", "StorageRetentionMode"]
