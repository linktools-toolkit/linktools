#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Persistent, user-editable JSON config store (spec §8.5 CFG-005).

Distinct from the cache: this holds state the user cares about and may edit by
hand (e.g. cntr's INSTALLED_CONTAINERS/INSTALLED_REPOS). The file is plain JSON,
written atomically under a process lock so concurrent writers never corrupt it
or lose each other's keys:

    acquire process lock -> reread -> modify -> temp -> fsync -> os.replace

A separate file (not the cache DB) so its lifecycle is independent, it is easy
to back up / version, and ``ct-cntr config edit`` edits something readable.
"""

import contextlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .errors import ConfigError
from .types import MISSING
from .utils import atomic_write

__all__ = ["ConfigStore"]


class ConfigStore(object):
    """A locked, atomically-written JSON key/value file."""

    def __init__(self, path, lock_manager=None):
        # type: (Any, Optional[Any]) -> None
        self._path = Path(str(path))
        self._lock_manager = lock_manager
        self._data = {}  # type: Dict[str, Any]
        self.reload()

    @property
    def path(self):
        # type: () -> Path
        return self._path

    # -- load / flush -------------------------------------------------------

    def reload(self):
        # type: () -> None
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

    def _flush(self):
        # type: () -> None
        atomic_write(
            self._path,
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
        )

    # -- locking ------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self):
        # type: () -> Iterator[None]
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

    def get(self, key, default=MISSING):
        # type: (str, Any) -> Any
        """Return the value for ``key``, or ``default`` if absent (v4 §3.4).

        Uses MISSING as the sentinel so stored None is distinguishable from
        a missing key (``key in store`` vs ``store.get(key) is None``).
        """
        if key in self._data:
            return self._data[key]
        return default

    def __contains__(self, key):
        # type: (str) -> bool
        return key in self._data

    def keys(self):
        # type: () -> List[str]
        return list(self._data.keys())

    def items(self):
        # type: () -> List[tuple]
        return list(self._data.items())

    # -- write (all go through the locked, atomic protocol) -----------------

    def set(self, key, value):
        # type: (str, Any) -> None
        with self._locked():
            self._data[key] = value
            self._flush()

    def save(self, **kwargs):
        # type: (**Any) -> None
        with self._locked():
            self._data.update(kwargs)
            self._flush()

    def remove(self, *keys):
        # type: (*str) -> bool
        removed = False
        with self._locked():
            for key in keys:
                if key in self._data:
                    self._data.pop(key, None)
                    removed = True
            self._flush()
        return removed

    def __repr__(self):
        # type: () -> str
        return "ConfigStore(path=%r, keys=%d)" % (str(self._path), len(self._data))
