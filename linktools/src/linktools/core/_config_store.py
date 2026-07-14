#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A locked, atomically-written JSON key/value file: the persistence layer
under ``PersistentSource`` / cntr's installed-container and repo state."""

import contextlib
import copy
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ConfigError
from ..types import MISSING
from ..utils import atomic_write

if TYPE_CHECKING:
    from typing import Any, Iterator

__all__ = ["ConfigStore", "ConfigNamespace"]


class ConfigStore(object):
    """A locked, atomically-written JSON key/value file."""

    def __init__(self, path: "Any", lock_manager: "Any | None" = None) -> None:
        self._path = Path(str(path))
        self._lock_manager = lock_manager
        self._data: "dict[str, Any]" = {}
        self._revision = 0
        self._tx_owner = threading.local()
        self.reload()

    @property
    def path(self) -> "Path":
        return self._path

    @property
    def revision(self) -> int:
        """Bumped on every successful reload/set/save/remove -- lets a
        PersistentSource wrapping this store (and anything comparing a
        cached revision token against it, e.g. ConfigResolver) detect a
        change made through a *different* PersistentSource/Config instance
        wrapping the same underlying file."""
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    # -- load / flush -------------------------------------------------------

    def reload(self) -> None:
        """Re-read the file; genuinely missing -> empty, anything else
        unreadable -> ConfigError (fail-closed, never a silent empty
        config: a dangling symlink or non-regular path must never be
        mistaken for "nothing configured")."""
        if not self._path.exists():
            if self._path.is_symlink():
                raise ConfigError("config store path is a dangling symlink: %s" % self._path)
            self._data = {}
            self._touch()
            return
        if not self._path.is_file():
            raise ConfigError("config store path is not a regular file: %s" % self._path)
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError("cannot read config %s: %s" % (self._path, exc))
        try:
            data = json.loads(text)
        except ValueError as exc:
            # User-editable file: surface the corruption rather than silently
            # wiping it on the next write.
            raise ConfigError("config %s is not valid JSON: %s" % (self._path, exc))
        if not isinstance(data, dict):
            raise ConfigError("config %s must be a JSON object, got %s" % (self._path, type(data).__name__))
        self._data = data
        self._touch()

    def _flush(self) -> None:
        atomic_write(
            self._path,
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
        )

    # -- locking ------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> "Iterator[None]":
        """Acquire the cross-process lock, reread, yield, then flush on exit.

        Reentrant for the thread already holding it (e.g. an ordinary set()
        called from within a namespace transaction on the same store): reused
        directly rather than acquired again, since a fresh ``FileLock``/
        ``process_lock`` object per call is not guaranteed OS-level reentrant
        on the same path, and re-reading here would discard whatever the
        outer call already has staged in memory. Safe to skip both: nobody
        else can be writing this file while this thread holds the lock.
        """
        if getattr(self._tx_owner, "value", None) == threading.get_ident():
            yield
            return
        if self._lock_manager is not None:
            lock = self._lock_manager.process_lock("config:" + self._path.name)
        else:
            # Fall back to a private filelock beside the config file.
            from filelock import FileLock

            lock = FileLock(str(self._path) + ".lock")
        with lock:
            self.reload()
            self._tx_owner.value = threading.get_ident()
            try:
                yield
            finally:
                self._tx_owner.value = None

    # -- read ---------------------------------------------------------------

    def get(self, key: str, default: "Any" = MISSING) -> "Any":
        """Return the value for ``key``, or ``default`` if absent.

        Uses MISSING as the sentinel so stored None is distinguishable from
        a missing key (``key in store`` vs ``store.get(key) is None``).
        """
        if key in self._data:
            return self._data[key]
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> "list[str]":
        return list(self._data.keys())

    def items(self) -> "list[tuple]":
        return list(self._data.items())

    # -- write (all go through the locked, atomic protocol) -----------------

    def set(self, key: str, value: "Any") -> None:
        with self._locked():
            previous = self._data
            self._data = dict(self._data)
            self._data[key] = value
            self._flush_or_rollback(previous)
            self._touch()

    def save(self, **kwargs: "Any") -> None:
        with self._locked():
            previous = self._data
            self._data = dict(self._data)
            self._data.update(kwargs)
            self._flush_or_rollback(previous)
            self._touch()

    def remove(self, *keys: str) -> bool:
        removed = False
        with self._locked():
            previous = self._data
            self._data = dict(self._data)
            for key in keys:
                if key in self._data:
                    self._data.pop(key, None)
                    removed = True
            if removed:
                self._flush_or_rollback(previous)
                self._touch()
        return removed

    def _flush_or_rollback(self, previous: "dict[str, Any]") -> None:
        """Write ``self._data`` to disk; on failure, restore ``previous``
        before re-raising -- a failed flush (disk full, permission error)
        must never leave the in-memory store reflecting a value that was
        never actually persisted, which a caller reading it back (without
        an intervening ``reload()``) would otherwise see as if it had
        succeeded."""
        try:
            self._flush()
        except Exception:
            self._data = previous
            raise

    def namespace(self, name: str) -> "ConfigNamespace":
        """A namespaced view over this store -- see :class:`ConfigNamespace`."""
        return ConfigNamespace(self, name)

    def __repr__(self) -> str:
        return "ConfigStore(path=%r, keys=%d)" % (str(self._path), len(self._data))


class ConfigNamespace(object):
    """A namespaced key/value view over one :class:`ConfigStore`.

    Mirrors ``CacheNamespace``'s ``get``/``set``/``pop``/``transaction``
    shape, for persistent (not swept, not TTL-expired) per-owner state --
    e.g. a container's own operational settings. Every namespace of one
    store shares its single JSON file; a namespace's data lives nested
    under one top-level key (its name) holding a dict.
    """

    def __init__(self, store: "ConfigStore", name: str) -> None:
        self._store = store
        self._name = name
        self._tx_data: "dict[str, Any] | None" = None  # set only inside transaction()

    @property
    def name(self) -> str:
        return self._name

    def _snapshot(self) -> "dict[str, Any]":
        if self._tx_data is not None:
            return self._tx_data
        # Deep copy: ConfigStore.get() returns a live reference into its own
        # _data, not a fresh decode like CacheNamespace's SQLite-blob reads --
        # a caller mutating a nested value in the returned dict must never
        # silently corrupt this store's in-memory state without going
        # through set()/a transaction.
        return copy.deepcopy(self._store.get(self._name, {}) or {})

    def get(self, key: str, default: "Any" = None) -> "Any":
        return self._snapshot().get(key, default)

    def keys(self) -> "list[str]":
        return list(self._snapshot().keys())

    def set(self, key: str, value: "Any") -> None:
        if self._tx_data is not None:
            self._tx_data[key] = value
            return
        with self.transaction():
            self.set(key, value)

    def pop(self, key: str, default: "Any" = None) -> "Any":
        if self._tx_data is not None:
            return self._tx_data.pop(key, default)
        with self.transaction():
            return self.pop(key, default)

    @contextlib.contextmanager
    def transaction(self) -> "Iterator[ConfigNamespace]":
        """Run a batch of get/set/pop against a consistent snapshot,
        flushing once on exit. Refuses to nest -- see ``ConfigStore._locked``."""
        with self._store._locked():
            self._tx_data = copy.deepcopy(self._store._data.get(self._name, {}) or {})
            previous = copy.deepcopy(self._tx_data)
            try:
                yield self
            except Exception:
                self._tx_data = None
                raise
            if self._tx_data != previous:
                previous_all = self._store._data
                new_all = dict(self._store._data)
                new_all[self._name] = self._tx_data
                self._store._data = new_all
                self._store._flush_or_rollback(previous_all)
                self._store._touch()
            self._tx_data = None

    def __repr__(self) -> str:
        return "ConfigNamespace(store=%r, name=%r)" % (self._store, self._name)
