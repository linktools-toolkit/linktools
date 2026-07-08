#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ToolInstaller: transactional managed-tool installation (spec §10.6/§10.8/§10.9).

STATUS: this module IS the install orchestrator for the Tools main path.
``Tool.prepare`` (in ``core/_tools.py``) delegates its install transaction to
``ToolInstaller.install_tool(tool)`` -- the download/extract/manifest/active/
lock/corrupt-move logic lives here, not duplicated in Tool. The on-disk layout
remains the Tool's own (config-resolved ``versions/<ver>-<plat>-<arch>/`` tree
+ ``active.json``); only the install *mechanism* is centralised (fix-plan §2.5:
no layout change). Behaviour is locked by ``tests/core/test_tools_prepare.py``.

The ``install(definition)`` method (ToolDefinition/registry-based, with the
``base/<name>/<version>/`` layout below) is the standalone/experimental path --
not wired into the business Tools flow. It is kept for a future layout
unification and is exercised only by its own unit tests.

Composes the persistence foundation: DownloadManager (download+validate),
utils.safe_extract (§10.7), LockManager (install lock), utils.atomic_write
(manifest + active pointer).

Layout (spec §10.9 multi-version)::

    base/<name>/<version>/        the installed tool tree + manifest.json
    base/<name>/active.json       points at the active version
    base/.corrupt/<name>-<ver>-<ts>  quarantined incomplete targets (§7.5)

Install flow (§7.4): acquire tool lock -> quarantine any corrupt target ->
staging/root -> download + validate -> safe_extract to empty dir -> validate
entrypoint in staging -> collect files -> write manifest inside staging ->
atomic move to <version> -> write active pointer -> cleanup. A failure at any
step removes staging; an already-complete version is reused (never re-extracted).
"""

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .. import utils
from .. import system as _system
from ..errors import ToolInstallError

if TYPE_CHECKING:
    from typing import Any

__all__ = ["ToolInstallation", "ToolInstaller"]

_MANIFEST_SCHEMA = 1

# safe_extract caps (spec §7.7): tighter than the generic defaults so a
# malicious / oversized archive cannot exhaust disk during a tool install.
_MAX_FILES = 20000
_MAX_TOTAL_SIZE = 2 * 1024 * 1024 * 1024      # 2 GiB
_MAX_SINGLE_FILE = 1 * 1024 * 1024 * 1024     # 1 GiB


class ToolInstallation(object):
    """An installed tool tree (spec §10.2 ToolInstallation)."""

    def __init__(self, root: "Path", name: str, version: str, manifest: dict) -> None:
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
    def __init__(self, environ: "Any", base_dir: "Any") -> None:
        self._environ = environ
        self._base = Path(str(base_dir))

    # -- internals ---------------------------------------------------------

    def _lock(self, name):
        # Unified tool-level lock (spec §7.6): install/remove/activate/gc/repair
        # all serialize on the same key.
        return self._environ.locks.process_lock("tool:" + name)

    def _version_dir(self, name: str, version: str) -> "Path":
        return self._base / name / version

    def _write_manifest(self, target, *, name, version, platform, architecture,
                        source_url, sha256, size, entrypoint, files):
        manifest = {
            "schema": _MANIFEST_SCHEMA,
            "name": name,
            "version": version,
            "platform": platform,
            "architecture": architecture,
            "source_url": source_url,
            "sha256": sha256,
            "size": size,
            "entrypoint": entrypoint,
            "installed_at": _now_iso(),
            "files": files,
        }
        utils.atomic_write(target / "manifest.json", json.dumps(manifest, indent=2))
        return manifest

    def _set_active(self, name, version):
        active = {"version": version}
        utils.atomic_write(self._base / name / "active.json", json.dumps(active))

    def _quarantine(self, target):
        """Move an incomplete/corrupt target aside so a fresh install can proceed."""
        corrupt_dir = self._base / ".corrupt"
        corrupt_dir.mkdir(parents=True, exist_ok=True)
        dest = corrupt_dir / ("%s-%s" % (target.name, _now_iso().replace(":", "")))
        try:
            shutil.move(str(target), str(dest))
        except OSError:
            shutil.rmtree(str(target), ignore_errors=True)

    # -- state (spec §7.3) ------------------------------------------------

    def is_installation_complete(self, name: str, version: str) -> bool:
        """Whether a version dir is a fully usable installation.

        Inspects only the version directory (manifest schema/name/version
        match, recorded files present, entrypoint present + executable on
        POSIX). Does NOT depend on the active pointer (§7.3.1).
        """
        target = self._version_dir(name, version)
        manifest_path = target / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (ValueError, OSError):
            return False
        if manifest.get("schema") != _MANIFEST_SCHEMA:
            return False
        if manifest.get("name") != name or manifest.get("version") != version:
            return False
        # Platform/architecture, when recorded, must match the current host.
        m_platform = manifest.get("platform")
        m_arch = manifest.get("architecture")
        if m_platform and m_platform != _system.get_system():
            return False
        if m_arch and m_arch != _system.normalize_arch(_system.get_machine()):
            return False
        # Recorded files must still exist.
        for rel in manifest.get("files", []):
            if not (target / rel).exists():
                return False
        # entrypoint must exist, live inside the install dir, and (on POSIX)
        # be executable.
        entrypoint = manifest.get("entrypoint")
        if entrypoint:
            ep_path = target / entrypoint if not os.path.isabs(entrypoint) else Path(entrypoint)
            if not ep_path.exists():
                return False
            if not utils.is_sub_path(str(ep_path), str(target)):
                return False
            if os.name != "nt" and not os.access(str(ep_path), os.X_OK):
                return False
        return True

    def is_active_valid(self, name: str) -> bool:
        """Whether the active pointer resolves to a complete installation (§7.3.2)."""
        active = self.active_version(name)
        if active is None:
            return False
        return self.is_installation_complete(name, active)

    def resolve_active(self, name: str) -> "ToolInstallation":
        """Return the active installation; raise if the active pointer is invalid."""
        active = self.active_version(name)
        if active is None or not self.is_installation_complete(name, active):
            raise ToolInstallError("no valid active installation for %s" % name)
        target = self._version_dir(name, active)
        manifest = json.loads((target / "manifest.json").read_text("utf-8"))
        return ToolInstallation(target, name, active, manifest)

    def is_installed(self, name: str, version: str) -> bool:
        """Backward-compatible alias for :meth:`is_installation_complete`."""
        return self.is_installation_complete(name, version)

    # -- install (spec §7.4) ----------------------------------------------

    def install(self, definition, *, source_url, sha256=None, version=None,
                downloader=None):
        """Install ``definition`` from ``source_url`` transactionally."""
        from .._download import DownloadRequest

        name = definition.name
        version = version or definition.version or "unknown"
        downloader = downloader or self._environ.downloads
        platform = _system.get_system()
        architecture = _system.normalize_arch(_system.get_machine())
        entrypoint = getattr(definition, "entrypoint", None)

        with self._lock(name):
            target = self._version_dir(name, version)
            if self.is_installation_complete(name, version):
                manifest = json.loads((target / "manifest.json").read_text("utf-8"))
                return ToolInstallation(target, name, version, manifest)

            # A half-written target blocks the atomic move; quarantine it so
            # the install can proceed from scratch (§7.5).
            if target.exists():
                self._quarantine(target)

            (self._base / name).mkdir(parents=True, exist_ok=True)
            staging = self._base / name / (".staging-%s" % uuid.uuid4().hex[:8])
            staging.mkdir(parents=True)
            try:
                archive = staging / "archive"
                request = DownloadRequest(url=source_url, destination=archive,
                                          sha256=sha256)
                downloader.download(request)

                extract_root = staging / "root"
                utils.safe_extract(archive, extract_root,
                                   max_files=_MAX_FILES,
                                   max_total_size=_MAX_TOTAL_SIZE,
                                   max_file_size=_MAX_SINGLE_FILE)
                archive.unlink()

                # Validate the entrypoint inside staging BEFORE the move (§7.4):
                # the target must never appear without a usable entrypoint.
                if entrypoint:
                    ep = extract_root / entrypoint
                    if not ep.exists():
                        raise ToolInstallError(
                            "entrypoint %r missing in archive for %s" % (entrypoint, name))

                files = sorted(p.relative_to(extract_root).as_posix()
                               for p in extract_root.rglob("*") if p.is_file())
                size = sum(p.stat().st_size for p in extract_root.rglob("*")
                           if p.is_file())
                manifest = self._write_manifest(
                    extract_root, name=name, version=version,
                    platform=platform, architecture=architecture,
                    source_url=source_url, sha256=sha256,
                    size=size, entrypoint=entrypoint, files=files)
                # Atomic move: target appears only when fully installed.
                target.parent.mkdir(parents=True, exist_ok=True)
                _move_tree(extract_root, target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

            self._set_active(name, version)
            return ToolInstallation(target, name, version, manifest)

    # -- main-path install: Tool delegation (fix-plan §3.3.3) -------------

    def install_tool(self, tool):
        """Install a (legacy) ``Tool`` object into its config-resolved layout.

        ``Tool.prepare`` delegates its install here so ToolInstaller is the
        single install orchestrator for the main path. The on-disk layout,
        manifest, and active.json format remain the Tool's own (the config-
        resolved ``versions/<ver>-<plat>-<arch>/`` tree) -- only the install
        *mechanism* is centralised here. Behaviour is identical to the previous
        in-line Tool.prepare block (locked by test_tools_prepare.py).
        """
        import uuid as _uuid
        from .._download import DownloadRequest

        env = tool._tools.environ
        name = tool.name
        with env.locks.process_lock("tool:" + name):
            if tool.exists:
                return  # another process installed it, or already present
            tool._tools.logger.info("Download %s: %s" % (tool, tool.download_url))
            temp_dir = env.get_temp_path("tools", "cache")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = str(temp_dir / utils.guess_file_name(tool.download_url))
            # DownloadManager owns atomic landing / resume / hash validation;
            # sha256/size are passed only when the tool definition declares them.
            env.downloads.download(DownloadRequest(
                url=tool.download_url, destination=temp_path,
                sha256=tool.get("sha256", None) or None,
                size=tool.get("size", None) or None,
            ))

            # staging dir -- everything happens here before the atomic move.
            staging = "%s.staging-%s" % (tool.root_path, _uuid.uuid4().hex[:8])
            os.makedirs(staging, exist_ok=True)
            corrupt = None
            try:
                if not utils.is_empty(tool.unpack_path):
                    tool._tools.logger.debug("Extract %s to %s" % (tool, staging))
                    utils.safe_extract(temp_path, staging)
                    os.remove(temp_path)
                else:
                    target_in_staging = os.path.join(
                        staging,
                        os.path.relpath(tool.absolute_path, tool.root_path))
                    os.makedirs(os.path.dirname(target_in_staging) or staging, exist_ok=True)
                    shutil.move(temp_path, target_in_staging)

                # manifest inside staging before the move (Tool's format).
                tool._write_manifest(staging)

                # Swap an existing (incomplete) root aside, atomically put
                # staging in place, restore on failure (fix-plan §2.3.2).
                if os.path.exists(tool.root_path):
                    corrupt = tool._make_corrupt_path(tool.root_path)
                    os.replace(tool.root_path, corrupt)
                try:
                    os.replace(staging, tool.root_path)
                    staging = None  # consumed
                except BaseException:
                    if not os.path.exists(tool.root_path) and corrupt \
                            and os.path.exists(corrupt):
                        os.replace(corrupt, tool.root_path)
                        corrupt = None
                    raise
                if corrupt:
                    shutil.rmtree(corrupt, ignore_errors=True)

                tool._set_active()
            except BaseException:
                if staging and os.path.exists(staging):
                    shutil.rmtree(staging, ignore_errors=True)
                if corrupt and os.path.exists(corrupt):
                    shutil.rmtree(corrupt, ignore_errors=True)
                raise

    def active_version(self, name: str) -> "str | None":
        pointer = self._base / name / "active.json"
        if not pointer.exists():
            return None
        try:
            return json.loads(pointer.read_text("utf-8")).get("version")
        except (ValueError, OSError):
            return None

    def remove(self, name: str, version: str, *, force: bool = False) -> bool:
        """Remove an installed version; refuse the active one unless ``force``."""
        with self._lock(name):  # unified lock (same as install)
            if not self.is_installation_complete(name, version):
                return False
            if not force and self.active_version(name) == version:
                raise ToolInstallError(
                    "refusing to remove active version %s of %s" % (version, name))
            _safe_remove_within(self._version_dir(name, version), self._base)
            return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _move_tree(src: "Path", dst: "Path") -> None:
    """Move ``src`` onto ``dst``; dst must not yet exist (atomic on same FS)."""
    if dst.exists():
        raise ToolInstallError("install target already exists: %s" % dst)
    os.replace(str(src), str(dst))


def _safe_remove_within(path: "Path", root: "Path") -> None:
    if not utils.is_sub_path(str(path), str(root)):
        raise ToolInstallError("refusing to remove %r outside %r" % (str(path), str(root)))
    shutil.rmtree(str(path), ignore_errors=True)


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
