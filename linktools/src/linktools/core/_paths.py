#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolved filesystem layout for an Environment (spec §5.3).

``EnvironmentPaths`` owns every directory linktools touches. Getters return
absolute, normalised :class:`~pathlib.Path` objects and **never create
directories** -- creation is explicit via the ``ensure_*`` methods. Deletes go
through :meth:`safe_remove`, which verifies the target is within an expected
root so a bad key can never escape the storage tree.

The layout (under a user ``storage`` root, plus a read-only package ``root``)::

    storage/
    +-- data/        persistent user data
    +-- temp/        ephemeral files
    +-- cache/       regenerable local cache (SQLite store lives here)
    +-- config/      editable persistent config (settings.json)
    +-- logs/        rotating log files
    +-- downloads/   download staging (.part files)
    +-- data/tools/  managed external-tool installations
    root/assets/     packaged read-only assets

Existing ``BaseEnviron`` accessors (``get_data_path`` etc.) keep working; this
class is the canonical model new code should obtain via ``environ.paths``.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union

from ..errors import EnvironmentError
from .. import utils

PathLike = Union[str, "os.PathLike[str]"]


def _norm(path: "PathLike") -> "Path":
    """Absolute, expanded, normalised -- matches utils.join_path semantics."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


class EnvironmentPaths(object):
    """Normalised directory layout for one :class:`Environment`."""

    def __init__(
        self,
        root: "PathLike",
        storage: "PathLike",
        *,
        data: "PathLike | None" = None,
        temp: "PathLike | None" = None,
        cache: "PathLike | None" = None,
        config: "PathLike | None" = None,
        tools: "PathLike | None" = None,
        downloads: "PathLike | None" = None,
        logs: "PathLike | None" = None,
        assets: "PathLike | None" = None,
        readonly: bool = False,
    ) -> None:
        self._root = _norm(root)
        self._storage = _norm(storage)
        self._data = _norm(data) if data is not None else self._storage / "data"
        self._temp = _norm(temp) if temp is not None else self._storage / "temp"
        self._cache = _norm(cache) if cache is not None else self._storage / "cache"
        self._config = _norm(config) if config is not None else self._storage / "config"
        self._logs = _norm(logs) if logs is not None else self._storage / "logs"
        self._downloads = _norm(downloads) if downloads is not None else self._storage / "downloads"
        # tools live under data (current convention); assets are packaged.
        self._tools = _norm(tools) if tools is not None else self._data / "tools"
        self._assets = _norm(assets) if assets is not None else self._root / "assets"
        self.readonly = bool(readonly)

    # -- getters (no side effects) ------------------------------------------

    @property
    def root(self) -> "Path":
        return self._root

    @property
    def storage(self) -> "Path":
        return self._storage

    @property
    def data(self) -> "Path":
        return self._data

    @property
    def temp(self) -> "Path":
        return self._temp

    @property
    def cache(self) -> "Path":
        return self._cache

    @property
    def config(self) -> "Path":
        return self._config

    @property
    def tools(self) -> "Path":
        return self._tools

    @property
    def downloads(self) -> "Path":
        return self._downloads

    @property
    def logs(self) -> "Path":
        return self._logs

    @property
    def assets(self) -> "Path":
        return self._assets

    # -- explicit creation --------------------------------------------------

    def ensure(self, path: "PathLike") -> "Path":
        """Create ``path`` (and parents) unless this layout is read-only.

        Returns the normalised path either way; read-only layouts skip creation
        so inspection never mutates the filesystem.
        """
        target = _norm(path)
        if not self.readonly:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def ensure_data(self) -> "Path":
        return self.ensure(self._data)

    def ensure_temp(self) -> "Path":
        return self.ensure(self._temp)

    def ensure_cache(self) -> "Path":
        return self.ensure(self._cache)

    def ensure_config(self) -> "Path":
        return self.ensure(self._config)

    def ensure_logs(self) -> "Path":
        return self.ensure(self._logs)

    def ensure_downloads(self) -> "Path":
        return self.ensure(self._downloads)

    def ensure_tools(self) -> "Path":
        return self.ensure(self._tools)

    # -- safe deletion (root-protected) -------------------------------------

    def safe_remove(self, path: "PathLike", root: "PathLike | None" = None) -> None:
        """Remove a file or directory after verifying it is within ``root``.

        ``root`` defaults to the storage tree. A target that resolves outside
        the root raises :class:`EnvironmentError` and is left untouched (spec
        §3.7 / §22.2: deletes must never escape the expected root).
        """
        target = _norm(path)
        boundary = _norm(root) if root is not None else self._storage
        if not utils.is_sub_path(str(target), str(boundary)):
            raise EnvironmentError(
                "Refusing to remove %r: outside root %r" % (str(target), str(boundary))
            )
        if not target.exists():
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                target.unlink()
            except FileNotFoundError:
                pass

    # -- test helper --------------------------------------------------------

    @classmethod
    def temporary(cls, prefix: str = "linktools-paths-") -> "EnvironmentPaths":
        """Build an isolated layout rooted at a fresh temp directory.

        ``root`` and ``storage`` share the temp directory; call ``cleanup()``
        when done. Intended for tests and embedded use.
        """
        storage = Path(tempfile.mkdtemp(prefix=prefix))
        paths = cls(root=storage, storage=storage)
        paths._cleanup_target = storage  # type: ignore[attr-defined]
        return paths

    def cleanup(self) -> None:
        """Remove the temp storage created by :meth:`temporary` (no-op otherwise)."""
        target = getattr(self, "_cleanup_target", None)
        if target is not None and target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def __repr__(self) -> str:
        return "EnvironmentPaths(storage=%r, root=%r, readonly=%r)" % (
            str(self._storage), str(self._root), self.readonly,
        )
