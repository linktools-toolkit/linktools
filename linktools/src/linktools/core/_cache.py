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
