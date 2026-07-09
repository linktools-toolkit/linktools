#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import json
import shelve as _shelve
import sqlite3
import threading
import time
import typing as _t
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import (
    CacheBackendError,
    CacheBusyError,
    CacheCodecError,
    CacheTransactionError,
    CacheValueError,
)
from .types import MISSING, MissingType, PathType, T, Timeout, TimeoutType

if TYPE_CHECKING:
    from typing import Any, Iterator

_basic_cache_types = (type(None), int, float, bool, complex)
_file_cache_backup_suffix = ".backup"


class _CacheLock:

    def __init__(self, cache: "FileCache", namespace: str, key: str = None):
        from filelock import FileLock

        path = cache.directory / namespace
        path.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(path / f"{key or ''}.lock"))

    def acquire(self, timeout: "TimeoutType" = None):
        self._lock.acquire(Timeout(timeout).remaining)

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

    def _get_version_path(self, version: str) -> "Path":
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
        # Store the full record unfiltered: falsy values (False/0/""/[]/{}) and
        # None must round-trip reliably, and existence is decided by row
        # presence rather than truthiness (spec  ttl=0 means "already
        # expired" and must not be coerced to None (= infinite) (spec 
        if ttl is not None:
            ttl = int(ttl)
        self._data[key] = {
            "data": self._cache.serialize(value),
            "ttl": ttl,
            "ts": int(time.time()),
        }

    def update(self, **kwargs: "T") -> None:
        for key, value in kwargs.items():
            self.set(key, value)

    def _get(self, key: str) -> "dict[str, _t.Any] | None":
        value = self._data.get(key, None)
        if value is not None:
            timestamp = value.get("ts", None)
            ttl = value.get("ttl", None)
            # Expiry is decided by presence (ttl is not None), never by
            # truthiness -- ttl=0 means "already expired" (spec 
            if timestamp is not None and ttl is not None \
                    and timestamp + ttl < time.time():
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
            value["ts"] = int(time.time())
            self._data[key] = value
        else:
            # Missing key: store initial + delta (not just the default), so the
            # first increment is never lost (spec 
            result = default + delta
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
        self._directory = Path(directory)
        self._serialize = serialize
        self._unserialize = unserialize

    @property
    def directory(self) -> "Path":
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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace   TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       BLOB    NOT NULL,
    codec       TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL,
    expires_at  REAL,
    version     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries(expires_at);
"""


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------

class CacheCodec(object):
    """(en|de)code cache values to/from bytes."""

    mime = "opaque"

    def encode(self, value: "Any") -> bytes:
        raise NotImplementedError

    def decode(self, blob: bytes) -> "Any":
        raise NotImplementedError


class JsonCodec(CacheCodec):
    mime = "json"

    def encode(self, value: "Any") -> bytes:
        try:
            return json.dumps(value, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CacheCodecError("value is not JSON-serialisable: %s" % exc)

    def decode(self, blob: bytes) -> "Any":
        try:
            return json.loads(blob.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CacheCodecError("value is not valid JSON: %s" % exc)


class BytesCodec(CacheCodec):
    """Pass-through codec for raw bytes only (no implicit pickling)."""

    mime = "bytes"

    def encode(self, value: "Any") -> bytes:
        if not isinstance(value, (bytes, bytearray)):
            raise CacheCodecError("BytesCodec only accepts bytes, got %s" % type(value).__name__)
        return bytes(value)

    def decode(self, blob: bytes) -> "Any":
        return bytes(blob)


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------

class CacheNamespace(object):
    """A key/value view over one :class:`CacheStore` namespace."""

    def __init__(self, store: "CacheStore", name: str, codec: "CacheCodec | None" = None) -> None:
        self._store = store
        self._name = name
        self._codec = codec or store.codec

    @property
    def name(self) -> str:
        return self._name

    # -- internals ---------------------------------------------------------

    def _conn(self):
        return self._store._conn()

    def _row(self, key: str) -> "sqlite3.Row | None":
        """Return the live row for ``key``, deleting+dropping it if expired."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM cache_entries WHERE namespace=? AND key=?",
            (self._name, key),
        ).fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if expires_at is not None and expires_at <= time.time():
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace=? AND key=?",
                (self._name, key),
            )
            return None
        return row

    def _decode(self, row: "sqlite3.Row") -> "Any":
        try:
            return self._codec.decode(row["value"])
        except CacheCodecError:
            raise
        except Exception as exc:  # defensive: codecs raise CacheCodecError, but be safe
            raise CacheCodecError("failed to decode %r: %s" % (row["key"], exc))

    # -- read --------------------------------------------------------------

    def get(self, key: str, default: "Any" = None) -> "Any":
        row = self._row(key)
        if row is None:
            return default
        return self._decode(row)

    def contains(self, key: str) -> bool:
        return self._row(key) is not None

    # -- write -------------------------------------------------------------

    # SQLite ≥3.24 supports ON CONFLICT … DO UPDATE (UPSERT).  Python 3.6 may
    # ship an older SQLite, so detect once and fall back to INSERT-or-UPDATE.
    _SUPPORTS_UPSERT = sqlite3.sqlite_version_info >= (3, 24, 0)

    def _exec_in_tx(self, conn, fn):
        """Execute fn inside a transaction; if one is already active, just run fn."""
        if getattr(self._store._tx_owner, "value", None) is not None:
            return fn(conn)
        self._begin(conn)
        try:
            result = fn(conn)
            conn.execute("COMMIT")
            return result
        except BaseException:
            self._rollback(conn)
            raise

    def set(self, key: str, value: "Any", ttl: "float | None" = None) -> None:
        expires_at = self._compute_expiry(ttl)
        blob = self._codec.encode(value)
        now = time.time()
        conn = self._conn()

        def _do_set(c):
            if CacheNamespace._SUPPORTS_UPSERT:
                c.execute(
                    "INSERT INTO cache_entries(namespace, key, value, codec, created_at,"
                    " updated_at, expires_at, version) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(namespace, key) DO UPDATE SET "
                    "value=excluded.value, codec=excluded.codec, "
                    "updated_at=excluded.updated_at, expires_at=excluded.expires_at, "
                    "version=cache_entries.version + 1",
                    (self._name, key, blob, self._codec.mime, now, now, expires_at),
                )
            else:
                cur = c.execute(
                    "UPDATE cache_entries SET value=?, codec=?, updated_at=?,"
                    " expires_at=?, version=version+1"
                    " WHERE namespace=? AND key=?",
                    (blob, self._codec.mime, now, expires_at, self._name, key),
                )
                if cur.rowcount == 0:
                    c.execute(
                        "INSERT INTO cache_entries(namespace, key, value, codec,"
                        " created_at, updated_at, expires_at, version)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?, 1)",
                        (self._name, key, blob, self._codec.mime, now, now, expires_at),
                    )

        self._exec_in_tx(conn, _do_set)

    @staticmethod
    def _compute_expiry(ttl: "float | None") -> "float | None":
        if ttl is None:
            return None
        ttl = float(ttl)
        if ttl < 0:
            raise CacheValueError("ttl must be non-negative, got %r" % (ttl,))
        return time.time() + ttl

    def delete(self, key: str) -> bool:
        conn = self._conn()
        def _do_delete(c):
            cur = c.execute(
                "DELETE FROM cache_entries WHERE namespace=? AND key=?",
                (self._name, key),
            )
            return cur.rowcount > 0
        return self._exec_in_tx(conn, _do_delete)

    def increment(self, key: str, delta: int = 1, initial: int = 0) -> int:
        """Atomically add ``delta`` (initial+delta when the key is absent).

        Goes through _exec_in_tx so it composes with an outer
        ``with namespace.transaction()`` instead of starting a nested tx.
        """
        conn = self._conn()

        def _do_increment(c):
            row = c.execute(
                "SELECT value FROM cache_entries WHERE namespace=? AND key=?",
                (self._name, key),
            ).fetchone()
            now = time.time()
            if row is None:
                result = initial + delta
                c.execute(
                    "INSERT INTO cache_entries(namespace, key, value, codec,"
                    " created_at, updated_at, expires_at, version) "
                    "VALUES(?, ?, ?, ?, ?, ?, NULL, 1)",
                    (self._name, key, self._codec.encode(result), self._codec.mime, now, now),
                )
            else:
                current = self._decode(row)
                if not isinstance(current, (int, float)) or isinstance(current, bool):
                    raise CacheValueError("value of %r is not numeric" % (key,))
                result = current + delta
                c.execute(
                    "UPDATE cache_entries SET value=?, updated_at=?, "
                    "version=version + 1 WHERE namespace=? AND key=?",
                    (self._codec.encode(result), now, self._name, key),
                )
            return result

        return self._exec_in_tx(conn, _do_increment)

    # -- iteration (snapshots,  ---------------------------------------

    def keys(self) -> "list[str]":
        return [k for k, _v in self._live_items()]

    def items(self) -> "list[tuple[str, Any]]":
        return self._live_items()

    def _live_items(self) -> "list[tuple[str, Any]]":
        conn = self._conn()
        rows = conn.execute(
            "SELECT key, value, expires_at FROM cache_entries WHERE namespace=? "
            "ORDER BY key",
            (self._name,),
        ).fetchall()
        out: "list[tuple[str, Any]]" = []
        now = time.time()
        for row in rows:
            expires_at = row["expires_at"]
            if expires_at is not None and expires_at <= now:
                continue  # expired; left for a cleanup pass / lazy _row
            out.append((row["key"], self._decode(row)))
        return out

    # -- transaction ( ------------------------------------------------

    @contextlib.contextmanager
    def transaction(self) -> "Iterator[CacheNamespace]":
        """Run a batch of set/delete atomically; roll back on any error."""
        conn = self._conn()
        if getattr(self._store._tx_owner, "value", None) is not None:
            raise CacheTransactionError("transactions cannot be nested")
        self._store._tx_owner.value = threading.get_ident()
        self._begin(conn)
        try:
            yield self
            conn.execute("COMMIT")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            self._store._tx_owner.value = None

    # -- begin/commit helpers ----------------------------------------------

    @staticmethod
    def _begin(conn):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise CacheBusyError("cache is locked by another writer: %s" % exc)
            raise CacheBackendError("cache begin failed: %s" % exc)

    @staticmethod
    def _rollback(conn):
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CacheStore(object):
    """A SQLite-backed cache database, opened lazily per thread."""

    def __init__(self, path: "Any", codec: "CacheCodec | None" = None) -> None:
        self.path = str(path)
        self.codec = codec or JsonCodec()
        self._tls = threading.local()
        self._tx_owner = threading.local()
        self._tx_owner.value = None
        self._init_db()

    def _conn(self) -> "sqlite3.Connection":
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass  # network FS / restricted env: fall back to default journal
            conn.execute("PRAGMA busy_timeout=10000")
            self._tls.conn = conn
        return conn

    def _init_db(self):
        self._conn().executescript(_SCHEMA)

    def namespace(self, name: str, codec: "CacheCodec | None" = None) -> "CacheNamespace":
        return CacheNamespace(self, name, codec=codec)

    def close(self) -> None:
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            conn.close()
            self._tls.conn = None

    def __enter__(self) -> "CacheStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
