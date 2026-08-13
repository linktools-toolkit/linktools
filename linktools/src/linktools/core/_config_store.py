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
from ._locks import LockManager

if TYPE_CHECKING:
    from typing import Any, Iterator

__all__ = ["ConfigStore", "ConfigNamespace"]


class ConfigStore(object):
    """A locked, atomically-written JSON key/value file."""

    def __init__(self, path: "Any", lock_manager: "Any | None" = None) -> None:
        self._path = Path(str(path))
        self._lock_manager = lock_manager or LockManager(self._path.parent / ".linktools-locks")
        self._data: "dict[str, Any]" = {}
        self._revision = 0
        self._tx_owner = threading.local()
        self.reload()

    @property
    def path(self) -> "Path":
        return self._path

    @property
    def revision(self) -> int:
        """Invalidation token for PersistentSource / ConfigResolver.

        Bumped on: explicit reload (always), successful set/save/remove,
        changed edit commit, and internal lock refresh when disk content
        differs from in-memory. No-op edit or unchanged-disk refresh do
        not bump.
        """
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    # -- load / flush -------------------------------------------------------

    def _read_data(self) -> "dict[str, Any]":
        """Read and validate the file without modifying ``_data`` or revision.

        Genuinely missing path -> empty dict. Dangling symlink, non-regular
        file, invalid JSON, or non-object root -> ConfigError (fail-closed).
        """
        if not self._path.exists():
            if self._path.is_symlink():
                raise ConfigError("config store path is a dangling symlink: %s" % self._path)
            return {}
        if not self._path.is_file():
            raise ConfigError("config store path is not a regular file: %s" % self._path)
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError("cannot read config %s: %s" % (self._path, exc))
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigError("config %s is not valid JSON: %s" % (self._path, exc))
        if not isinstance(data, dict):
            raise ConfigError("config %s must be a JSON object, got %s" % (self._path, type(data).__name__))
        return data

    def _refresh_if_changed(self) -> None:
        """Internal lock refresh: only bump revision if disk differs."""
        data = self._read_data()
        if data != self._data:
            self._data = data
            self._touch()

    def reload(self) -> None:
        """Re-read the file (always bumps revision on success).

        Fail-closed: a dangling symlink or non-regular path raises
        ConfigError rather than silently presenting an empty config.
        """
        self._data = self._read_data()
        self._touch()

    def _flush(self) -> None:
        atomic_write(
            self._path,
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
        )

    # -- locking ------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> "Iterator[None]":
        """Acquire the cross-process lock and refresh from disk.

        Reentrant for the thread already holding it (e.g. set() called
        from within a namespace transaction on the same store). The
        reentrant branch skips both lock acquisition and disk refresh.
        """
        if getattr(self._tx_owner, "value", None) == threading.get_ident():
            yield
            return
        lock = self._lock_manager.file_lock(self._path)
        with lock:
            self._refresh_if_changed()
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
            self._flush_or_recover(previous)
            self._touch()

    def save(self, **kwargs: "Any") -> None:
        with self._locked():
            previous = self._data
            self._data = dict(self._data)
            self._data.update(kwargs)
            self._flush_or_recover(previous)
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
                self._flush_or_recover(previous)
                self._touch()
        return removed

    def _flush_or_recover(self, previous: "dict[str, Any]") -> None:
        """Write ``self._data`` to disk; on failure, synchronise memory to
        the best-known persisted state and re-raise the original exception.

        If ``_flush()`` fails before ``os.replace`` completes, the disk
        still holds ``previous`` — memory is restored to ``previous``. If
        the failure occurs after replace (e.g. a post-replace signal),
        the disk may already hold the new value — memory is synchronised
        to what the disk actually contains. If readback itself fails,
        memory falls back to ``previous``. The original flush exception
        always propagates.
        """
        try:
            self._flush()
        except BaseException:
            try:
                persisted = self._read_data()
            except BaseException:
                self._data = previous
            else:
                self._data = persisted
                if persisted != previous:
                    self._touch()
            raise

    def namespace(self, name: str) -> "ConfigNamespace":
        """A namespaced view over this store -- see :class:`ConfigNamespace`."""
        return ConfigNamespace(self, name)

    @contextlib.contextmanager
    def edit(self, key: str, default: "Any" = None) -> "Iterator[Any]":
        """Yield a deep-copy snapshot of ``key``'s value for in-place editing,
        committing atomically on a clean exit.

        The cross-process lock spans the entire read-edit-commit lifecycle.
        If the context body raises, the local staged edit is not committed.
        However, if the internal lock refresh detected and loaded external
        disk changes before the body ran, those external changes are not
        rolled back — only the local edit is discarded.

        A value equal to the pre-edit snapshot (deep equality) is a no-op:
        no flush, no revision bump.
        """
        with self._locked():
            if key in self._data:
                value = copy.deepcopy(self._data[key])
            else:
                value = copy.deepcopy(default)
            previous = copy.deepcopy(value)
            yield value
            if value != previous:
                previous_all = self._data
                new_all = dict(self._data)
                new_all[key] = value
                self._data = new_all
                self._flush_or_recover(previous_all)
                self._touch()

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
        flushing once on exit. Refuses to nest."""
        if self._tx_data is not None:
            raise ConfigError("config namespace transactions cannot be nested")
        with self._store.edit(self._name, {}) as data:
            self._tx_data = data
            try:
                yield self
            finally:
                self._tx_data = None

    def __repr__(self) -> str:
        return "ConfigNamespace(store=%r, name=%r)" % (self._store, self._name)
