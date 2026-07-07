#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ToolInstaller: transactional managed-tool installation (spec §10.6/§10.8/§10.9).

Standalone build (Phase 5 PR 11). Composes the persistence foundation:
DownloadManager (download+validate), utils.safe_extract (§10.7), LockManager
(install lock), utils.atomic_write (manifest + active pointer). The legacy
Tool.prepare (core/_tools.py) stays until consumers migrate.

Layout (spec §10.9 multi-version)::

    base/<name>/<version>/        the installed tool tree + manifest.json
    base/<name>/active.json       points at the active version

Install flow (§10.6): acquire install lock -> staging dir -> download + validate
-> safe extract -> write manifest -> atomic rename to <version> -> write active
pointer. A failure at any step removes staging; an already-installed version is
reused (never re-extracted).
"""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from .. import utils
from ..errors import ToolInstallError

__all__ = ["ToolInstallation", "ToolInstaller"]

_MANIFEST_SCHEMA = 1


class ToolInstallation(object):
    """An installed tool tree (spec §10.2 ToolInstallation)."""

    def __init__(self, root, name, version, manifest):
        # type: (Path, str, str, dict) -> None
        self.root = root
        self.name = name
        self.version = version
        self.manifest = manifest

    @property
    def path(self):
        return self.root

    def __repr__(self):
        return "ToolInstallation(%s/%s at %s)" % (self.name, self.version, self.root)


class ToolInstaller(object):
    def __init__(self, environ, base_dir):
        # type: (Any, Any) -> None
        self._environ = environ
        self._base = Path(str(base_dir))

    # -- internals ---------------------------------------------------------

    def _lock(self, name):
        return self._environ.locks.process_lock("tool-install:" + name)

    def _version_dir(self, name, version):
        # type: (str, str) -> Path
        return self._base / name / version

    def _write_manifest(self, target, *, name, version, source_url, sha256, files):
        manifest = {
            "schema": _MANIFEST_SCHEMA,
            "name": name,
            "version": version,
            "source_url": source_url,
            "sha256": sha256,
            "installed_at": _now_iso(),
            "files": files,
        }
        utils.atomic_write(target / "manifest.json", json.dumps(manifest, indent=2))
        return manifest

    def _set_active(self, name, version):
        active = {"version": version}
        utils.atomic_write(self._base / name / "active.json", json.dumps(active))

    # -- public ------------------------------------------------------------

    def is_installed(self, name, version):
        # type: (str, str) -> bool
        """Check whether a version is fully installed (v2 §8.5).

        Not just manifest existence: validates schema, name/version match, and
        that the active pointer is valid.
        """
        target = self._version_dir(name, version)
        manifest_path = target / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (ValueError, OSError):
            return False
        # Manifest schema + name/version match.
        if manifest.get("schema") != _MANIFEST_SCHEMA:
            return False
        if manifest.get("name") != name or manifest.get("version") != version:
            return False
        # v2 §8.5: active pointer must be valid (exists + points at an installed
        # version), but the version being checked does NOT have to be active.
        active = self.active_version(name)
        if active is None:
            return False
        if not (self._base / name / active / "manifest.json").exists():
            return False
        # Listed files must still exist.
        for rel in manifest.get("files", []):
            if not (target / rel).exists():
                return False
        return True

    def install(self, definition, *, source_url, sha256=None, version=None,
                downloader=None):
        """Install ``definition`` from ``source_url`` transactionally (§10.6).

        Downloads (validating ``sha256``) via ``downloader`` (defaults to
        environ.downloads), safe-extracts into a staging dir, writes a manifest,
        then atomically renames to the version dir and updates the active
        pointer. Reuses an already-installed version.
        """
        from .._download import DownloadRequest

        name = definition.name
        version = version or definition.version or "unknown"
        downloader = downloader or self._environ.downloads

        with self._lock(name):
            target = self._version_dir(name, version)
            if self.is_installed(name, version):
                manifest = json.loads((target / "manifest.json").read_text("utf-8"))
                return ToolInstallation(target, name, version, manifest)

            (self._base / name).mkdir(parents=True, exist_ok=True)
            staging = self._base / name / (".staging-%s" % uuid.uuid4().hex[:8])
            staging.mkdir(parents=True)
            try:
                archive = staging / "archive"
                request = DownloadRequest(url=source_url, destination=archive,
                                          sha256=sha256)
                downloader.download(request)

                # Safe extract into staging (§10.7): no traversal, no symlinks.
                extract_root = staging / "root"
                utils.safe_extract(archive, extract_root)
                archive.unlink()

                files = sorted(p.relative_to(extract_root).as_posix()
                               for p in extract_root.rglob("*") if p.is_file())
                # Move the extracted tree into place under the version dir.
                target.parent.mkdir(parents=True, exist_ok=True)
                _move_tree(extract_root, target)
                manifest = self._write_manifest(
                    target, name=name, version=version,
                    source_url=source_url, sha256=sha256, files=files)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                raise
            finally:
                # staging was renamed away on success; remove any leftover.
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

            self._set_active(name, version)
            return ToolInstallation(target, name, version, manifest)

    def active_version(self, name):
        # type: (str) -> Optional[str]
        pointer = self._base / name / "active.json"
        if not pointer.exists():
            return None
        return json.loads(pointer.read_text("utf-8")).get("version")

    def remove(self, name, version, *, force=False):
        # type: (str, str, bool) -> bool
        """Remove an installed version; refuse the active one unless ``force``."""
        with self._lock(name + ":remove"):
            if not self.is_installed(name, version):
                return False
            if not force and self.active_version(name) == version:
                raise ToolInstallError(
                    "refusing to remove active version %s of %s" % (version, name))
            target = self._version_dir(name, version)
            _safe_remove_within(target, self._base)
            return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _move_tree(src, dst):
    # type: (Path, Path) -> None
    """Move ``src`` onto ``dst``; dst must not yet exist (atomic on same FS)."""
    if dst.exists():
        raise ToolInstallError("install target already exists: %s" % dst)
    os.replace(str(src), str(dst))


def _safe_remove_within(path, root):
    # type: (Path, Path) -> None
    if not utils.is_sub_path(str(path), str(root)):
        raise ToolInstallError("refusing to remove %r outside %r" % (str(path), str(root)))
    shutil.rmtree(str(path), ignore_errors=True)


def _now_iso():
    # type: () -> str
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
