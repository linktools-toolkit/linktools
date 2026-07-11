#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A locked, atomically-written JSON key/value file: the persistence layer
under ``PersistentSource`` / cntr's installed-container and repo state."""

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import ConfigError
from ..types import MISSING
from ..utils import atomic_write

if TYPE_CHECKING:
    from typing import Any, Iterator

__all__ = ["ConfigStore"]


class ConfigStore(object):
    """A locked, atomically-written JSON key/value file."""

    def __init__(self, path: "Any", lock_manager: "Any | None" = None) -> None:
        self._path = Path(str(path))
        self._lock_manager = lock_manager
        self._data: "dict[str, Any]" = {}
        self.reload()

    @property
    def path(self) -> "Path":
        return self._path

    # -- load / flush -------------------------------------------------------

    def reload(self) -> None:
        """Re-read the file; missing -> empty, corrupt -> ConfigError."""
        if not self._path.exists():
            self._data = {}
            return
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

    def _flush(self) -> None:
        atomic_write(
            self._path,
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
        )

    # -- locking ------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> "Iterator[None]":
        """Acquire the cross-process lock, reread, yield, then flush on exit."""
        if self._lock_manager is not None:
            lock = self._lock_manager.process_lock("config:" + self._path.name)
        else:
            # Fall back to a private filelock beside the config file.
            from filelock import FileLock

            lock = FileLock(str(self._path) + ".lock")
        with lock:
            self.reload()
            yield

    # -- read ---------------------------------------------------------------

    def get(self, key: str, default: "Any" = MISSING) -> "Any":
        """Return the value for ``key``, or ``default`` if absent (v4 §3.4).

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
            self._data[key] = value
            self._flush()

    def save(self, **kwargs: "Any") -> None:
        with self._locked():
            self._data.update(kwargs)
            self._flush()

    def remove(self, *keys: str) -> bool:
        removed = False
        with self._locked():
            for key in keys:
                if key in self._data:
                    self._data.pop(key, None)
                    removed = True
            self._flush()
        return removed

    def __repr__(self) -> str:
        return "ConfigStore(path=%r, keys=%d)" % (str(self._path), len(self._data))
