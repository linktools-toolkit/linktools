#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import contextlib
import json
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

from ..errors import (
    CacheBackendError,
    CacheBusyError,
    CacheCodecError,
    CacheTransactionError,
    CacheValueError,
)

if TYPE_CHECKING:
    from typing import Any, Iterator

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

# SQLite >=3.24 supports ON CONFLICT ... DO UPDATE (UPSERT).
# Python 3.6 may ship an older SQLite, so detect once and fall back.
_SUPPORTS_UPSERT = sqlite3.sqlite_version_info >= (3, 24, 0)


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
# Store -- owns the SQLite backend and all transaction state
# ---------------------------------------------------------------------------

class CacheStore(object):
    """A SQLite-backed cache database, opened lazily per thread.

    Owns connection management, SQL execution, and transaction lifecycle.
    Namespaces are thin codec-scoped views that delegate here.
    """

    def __init__(self, path: "Any", codec: "CacheCodec | None" = None) -> None:
        self.path = str(path)
        self.codec = codec or JsonCodec()
        self._tls = threading.local()
        self._tx_owner = threading.local()
        self._tx_owner.value = None
        self._init_db()

    # -- connection --------------------------------------------------------

    def _conn(self) -> "sqlite3.Connection":
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            conn.execute("PRAGMA busy_timeout=10000")
            self._tls.conn = conn
        return conn

    def _init_db(self) -> None:
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

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -- row helpers -------------------------------------------------------

    def _live_row(self, conn, namespace: str, key: str) -> "sqlite3.Row | None":
        """Return the live row for ``(namespace, key)`` on ``conn``, deleting it if expired."""
        row = conn.execute(
            "SELECT * FROM cache_entries WHERE namespace=? AND key=?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        expires_at = row["expires_at"]
        if expires_at is not None and expires_at <= time.time():
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace=? AND key=?",
                (namespace, key),
            )
            return None
        return row

    def _row(self, namespace: str, key: str) -> "sqlite3.Row | None":
        return self._live_row(self._conn(), namespace, key)

    @staticmethod
    def _decode_row(row: "sqlite3.Row", codec: "CacheCodec") -> "Any":
        try:
            return codec.decode(row["value"])
        except CacheCodecError:
            raise
        except Exception as exc:
            raise CacheCodecError("failed to decode %r: %s" % (row["key"], exc))

    # -- transaction state -------------------------------------------------

    def _in_transaction(self) -> bool:
        return getattr(self._tx_owner, "value", None) is not None

    def _exec_in_tx(self, conn, fn):
        """Execute fn inside a transaction; if one is already active, just run fn."""
        if self._in_transaction():
            return fn(conn)
        self._begin(conn)
        try:
            result = fn(conn)
            conn.execute("COMMIT")
            return result
        except BaseException:
            self._rollback(conn)
            raise

    @staticmethod
    def _begin(conn) -> None:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise CacheBusyError("cache is locked by another writer: %s" % exc)
            raise CacheBackendError("cache begin failed: %s" % exc)

    @staticmethod
    def _rollback(conn) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    # -- public data API ---------------------------------------------------

    def get(self, namespace: str, key: str, codec: "CacheCodec | None" = None, default: "Any" = None) -> "Any":
        codec = codec or self.codec
        row = self._row(namespace, key)
        if row is None:
            return default
        return self._decode_row(row, codec)

    def contains(self, namespace: str, key: str) -> bool:
        return self._row(namespace, key) is not None

    def set(self, namespace: str, key: str, value: "Any", codec: "CacheCodec | None" = None, ttl: "float | None" = None) -> None:
        codec = codec or self.codec
        expires_at = self._compute_expiry(ttl)
        blob = codec.encode(value)
        now = time.time()
        conn = self._conn()

        def _do_set(c):
            if _SUPPORTS_UPSERT:
                c.execute(
                    "INSERT INTO cache_entries(namespace, key, value, codec, created_at,"
                    " updated_at, expires_at, version) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, 1) "
                    "ON CONFLICT(namespace, key) DO UPDATE SET "
                    "value=excluded.value, codec=excluded.codec, "
                    "updated_at=excluded.updated_at, expires_at=excluded.expires_at, "
                    "version=cache_entries.version + 1",
                    (namespace, key, blob, codec.mime, now, now, expires_at),
                )
            else:
                cur = c.execute(
                    "UPDATE cache_entries SET value=?, codec=?, updated_at=?,"
                    " expires_at=?, version=version+1"
                    " WHERE namespace=? AND key=?",
                    (blob, codec.mime, now, expires_at, namespace, key),
                )
                if cur.rowcount == 0:
                    c.execute(
                        "INSERT INTO cache_entries(namespace, key, value, codec,"
                        " created_at, updated_at, expires_at, version)"
                        " VALUES(?, ?, ?, ?, ?, ?, ?, 1)",
                        (namespace, key, blob, codec.mime, now, now, expires_at),
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

    def delete(self, namespace: str, key: str) -> bool:
        conn = self._conn()

        def _do_delete(c):
            cur = c.execute(
                "DELETE FROM cache_entries WHERE namespace=? AND key=?",
                (namespace, key),
            )
            return cur.rowcount > 0

        return self._exec_in_tx(conn, _do_delete)

    def increment(self, namespace: str, key: str, delta: int = 1, initial: int = 0, codec: "CacheCodec | None" = None) -> int:
        codec = codec or self.codec
        conn = self._conn()

        def _do_increment(c):
            row = self._live_row(c, namespace, key)
            now = time.time()
            if row is None:
                result = initial + delta
                c.execute(
                    "INSERT INTO cache_entries(namespace, key, value, codec,"
                    " created_at, updated_at, expires_at, version) "
                    "VALUES(?, ?, ?, ?, ?, ?, NULL, 1)",
                    (namespace, key, codec.encode(result), codec.mime, now, now),
                )
            else:
                current = self._decode_row(row, codec)
                if not isinstance(current, (int, float)) or isinstance(current, bool):
                    raise CacheValueError("value of %r is not numeric" % (key,))
                result = current + delta
                c.execute(
                    "UPDATE cache_entries SET value=?, updated_at=?, "
                    "version=version + 1 WHERE namespace=? AND key=?",
                    (codec.encode(result), now, namespace, key),
                )
            return result

        return self._exec_in_tx(conn, _do_increment)

    def keys(self, namespace: str) -> "list[str]":
        return [k for k, _v in self._live_items(namespace)]

    def items(self, namespace: str, codec: "CacheCodec | None" = None) -> "list[tuple[str, Any]]":
        codec = codec or self.codec
        return self._live_items(namespace, codec=codec)

    def _live_items(self, namespace: str, codec: "CacheCodec | None" = None) -> "list[tuple[str, Any]]":
        codec = codec or self.codec
        conn = self._conn()
        rows = conn.execute(
            "SELECT key, value, expires_at FROM cache_entries WHERE namespace=? "
            "ORDER BY key",
            (namespace,),
        ).fetchall()
        out: "list[tuple[str, Any]]" = []
        now = time.time()
        for row in rows:
            expires_at = row["expires_at"]
            if expires_at is not None and expires_at <= now:
                continue
            out.append((row["key"], self._decode_row(row, codec)))
        return out

    @contextlib.contextmanager
    def transaction(self) -> "Iterator[CacheStore]":
        """Run a batch of set/delete atomically; roll back on any error."""
        conn = self._conn()
        if self._in_transaction():
            raise CacheTransactionError("transactions cannot be nested")
        self._begin(conn)
        self._tx_owner.value = threading.get_ident()
        try:
            yield self
            conn.execute("COMMIT")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            self._tx_owner.value = None


# ---------------------------------------------------------------------------
# Namespace -- thin codec-scoped view over CacheStore
# ---------------------------------------------------------------------------

class CacheNamespace(object):
    """A key/value view over one :class:`CacheStore` namespace.

    Delegates all backend operations (SQL, transactions) to the store,
    supplying only the namespace name and default codec.
    """

    def __init__(self, store: "CacheStore", name: str, codec: "CacheCodec | None" = None) -> None:
        self._store = store
        self._name = name
        self._codec = codec or store.codec

    @property
    def name(self) -> str:
        return self._name

    def get(self, key: str, default: "Any" = None) -> "Any":
        return self._store.get(self._name, key, codec=self._codec, default=default)

    def contains(self, key: str) -> bool:
        return self._store.contains(self._name, key)

    def set(self, key: str, value: "Any", ttl: "float | None" = None) -> None:
        self._store.set(self._name, key, value, codec=self._codec, ttl=ttl)

    def delete(self, key: str) -> bool:
        return self._store.delete(self._name, key)

    def increment(self, key: str, delta: int = 1, initial: int = 0) -> int:
        return self._store.increment(self._name, key, delta=delta, initial=initial, codec=self._codec)

    def keys(self) -> "list[str]":
        return self._store.keys(self._name)

    def items(self) -> "list[tuple[str, Any]]":
        return self._store.items(self._name, codec=self._codec)

    @contextlib.contextmanager
    def transaction(self) -> "Iterator[CacheNamespace]":
        with self._store.transaction():
            yield self

    def __repr__(self) -> str:
        return "CacheNamespace(store=%r, name=%r)" % (self._store, self._name)
