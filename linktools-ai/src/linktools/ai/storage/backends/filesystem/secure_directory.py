#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecureDirectory: a dirfd-relative filesystem access scope.

The trusted root is opened once, and every operation descends component-by-
component via ``os.open(component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
dir_fd=parent)`` so a symlink planted anywhere in the chain (the target, an
ancestor, or a sibling swapped in mid-walk) is rejected by the kernel BEFORE
the read/write/delete follows it. This is the structural replacement for the
old "lstat-walk then re-resolve by Path" pattern (resolve_secure_path), which
checked the chain and then used the result on a freshly-resolved Path -- a
TOCTOU a privileged attacker could win.

Public methods take ``*components`` (each a single path segment, no slashes);
the class never accepts a Path or a multi-segment string from production code.
All fd lifecycle is bounded by context managers; every ``finally`` closes
every fd it opened, and every mutation fsyncs the parent directory so the
change is durable across a crash.

Platform capability: SECURE_POSIX (the only safe default) requires the kernel
to expose ``O_NOFOLLOW``, ``O_DIRECTORY``, ``O_CLOEXEC``, dir_fd-relative
``open``/``replace``/``unlink``/``mkdir``/``rmdir``/``stat``, and ``flock``.
Construction probes for these once; a platform that lacks any of them refuses
SECURE_POSIX (the caller must opt into ``TRUSTED_LOCAL`` explicitly and own
the consequences)."""

from __future__ import annotations

import json
import os
import secrets
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

from ...object.errors import StorageObjectError


class FilesystemSecurityMode(str, Enum):
    """SECURE_POSIX is the only safe default: every component is opened with
    O_NOFOLLOW|O_DIRECTORY through a dirfd chain, so a symlink planted
    anywhere is rejected by the kernel. TRUSTED_LOCAL must be opted into
    explicitly (e.g. a single-tenant local-only deployment that does not need
    the defense); it is refused when ``RuntimeSettings.multi_tenant=True``."""

    SECURE_POSIX = "secure_posix"
    TRUSTED_LOCAL = "trusted_local"


def _check_posix_capabilities() -> None:
    """Probe the platform for the dirfd/O_NOFOLLOW/O_DIRECTORY/flock primitives
    SECURE_POSIX depends on. Raises StorageObjectError if any are missing so
    a platform that silently degrades to path-based access cannot ship."""
    missing: "list[str]" = []
    for capability in ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC"):
        if not hasattr(os, capability):
            missing.append(f"os.{capability}")
    # dir_fd support on the operations we use:
    for fn_name in ("open", "mkdir", "rmdir", "unlink", "stat"):
        fn = getattr(os, fn_name, None)
        # CPython exposes dir_fd acceptance via the function's __doc__ /
        # signature; the reliable probe is to inspect support_os_dir_fd.
        if fn is None or fn not in getattr(os, "supports_dir_fd", set()):
            missing.append(f"os.{fn_name}")
    if not getattr(os, "replace", None) or os.rename not in getattr(os, "supports_dir_fd", set()):
        missing.append("os.replace (dir_fd form)")
    try:
        import fcntl  # noqa: F401
    except ImportError:
        missing.append("fcntl (flock)")
    else:
        if not hasattr(fcntl, "flock"):
            missing.append("fcntl.flock")
    if missing:
        raise StorageObjectError(
            "platform lacks SECURE_POSIX capabilities required for safe "
            f"dirfd-based filesystem access: {', '.join(missing)}"
        )


def _validate_components(components: "tuple[str, ...]") -> None:
    """Pure lexical check (no filesystem access): every component must be a
    single non-empty segment, with no slash, no ``.``/``..``, no NUL."""
    for part in components:
        if not part:
            raise StorageObjectError("empty path component not allowed")
        if part in (".", ".."):
            raise StorageObjectError(f"path traversal not allowed: {part!r}")
        if "/" in part:
            raise StorageObjectError(
                f"slash not allowed in a single component: {part!r}"
            )
        if "\x00" in part:
            raise StorageObjectError("NUL byte not allowed in path component")


class SecureDirectory:
    """A dirfd-scoped subtree rooted at ``root``. The root is opened on every
    operation (no cached fd), and the descent walks component-by-component
    with O_NOFOLLOW|O_DIRECTORY so a symlink anywhere in the chain is rejected
    by the kernel before the target fd is handed back. All write paths
    fsync the file AND the parent directory before returning."""

    def __init__(
        self,
        root: Path,
        *,
        mode: FilesystemSecurityMode = FilesystemSecurityMode.SECURE_POSIX,
    ) -> None:
        if mode is FilesystemSecurityMode.SECURE_POSIX:
            _check_posix_capabilities()
            self._dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            self._file_read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            self._file_create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._mode = mode

    @property
    def root(self) -> Path:
        return self._root

    @property
    def security_mode(self) -> FilesystemSecurityMode:
        return self._mode

    # --- internal: dirfd descent -------------------------------------------

    @contextmanager
    def _open_root(self) -> "Iterator[int]":
        fd = os.open(str(self._root), self._dir_flags)
        try:
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def _open_dir_chain(self, *components: str) -> "Iterator[int]":
        """Open the root, then each ``components`` as a directory (O_NOFOLLOW
        | O_DIRECTORY) relative to the previous one. Intermediate fds are
        closed as we descend; the deepest directory's fd is yielded. The
        components MUST all exist as directories. The root fd is owned by
        ``_open_root`` and is never closed here."""
        _validate_components(components)
        with self._open_root() as root_fd:
            if not components:
                yield root_fd
                return
            current = os.open(components[0], self._dir_flags, dir_fd=root_fd)
            try:
                for component in components[1:]:
                    next_fd = os.open(component, self._dir_flags, dir_fd=current)
                    os.close(current)
                    current = next_fd
                yield current
            finally:
                os.close(current)

    # --- public API ---------------------------------------------------------

    def ensure_directory(self, *components: str, mode: int = 0o700) -> None:
        """Idempotently create the directory ``root/components`` and every
        intermediate. Each newly created directory is fsync'd against its
        parent so the new entry is durable."""
        _validate_components(components)
        with self._open_root() as root_fd:
            if not components:
                return
            try:
                try:
                    current = os.open(components[0], self._dir_flags, dir_fd=root_fd)
                except FileNotFoundError:
                    os.mkdir(components[0], mode, dir_fd=root_fd)
                    os.fsync(root_fd)
                    current = os.open(components[0], self._dir_flags, dir_fd=root_fd)
                for component in components[1:]:
                    try:
                        next_fd = os.open(component, self._dir_flags, dir_fd=current)
                    except FileNotFoundError:
                        os.mkdir(component, mode, dir_fd=current)
                        os.fsync(current)
                        next_fd = os.open(component, self._dir_flags, dir_fd=current)
                    os.close(current)
                    current = next_fd
            finally:
                os.close(current)

    def read_bytes(self, *components: str) -> bytes:
        if not components:
            raise StorageObjectError("read_bytes requires at least one component")
        _validate_components(components)
        parent_components = components[:-1]
        name = components[-1]
        with self._open_dir_chain(*parent_components) as parent_fd:
            file_fd = os.open(name, self._file_read_flags, dir_fd=parent_fd)
            try:
                chunks: "list[bytes]" = []
                while True:
                    chunk = os.read(file_fd, 64 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(file_fd)

    def read_json(self, *components: str) -> "Mapping[str, Any]":
        return json.loads(self.read_bytes(*components).decode("utf-8"))

    def stat(self, *components: str) -> "os.stat_result | None":
        if not components:
            with self._open_root() as fd:
                return os.fstat(fd)
        _validate_components(components)
        parent_components = components[:-1]
        name = components[-1]
        try:
            with self._open_dir_chain(*parent_components) as parent_fd:
                try:
                    return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    return None
        except FileNotFoundError:
            # An intermediate directory does not exist; the target cannot
            # exist either.
            return None

    def list_names(self, *components: str) -> "tuple[str, ...]":
        try:
            with self._open_dir_chain(*components) as dir_fd:
                return tuple(sorted(os.listdir(dir_fd)))
        except FileNotFoundError:
            # A missing directory lists as empty.
            return ()

    def atomic_write(
        self,
        *components: str,
        content: bytes,
        mode: int = 0o600,
    ) -> None:
        """Atomically write ``content`` to ``root/components``. The bytes land
        in a uniquely-named temp file in the SAME directory as the target
        (O_CREAT|O_EXCL|O_NOFOLLOW), are fsync'd, then ``os.replace``'d onto
        the target name (atomic on POSIX), then the parent dir is fsync'd. On
        any exception (including cancellation) the temp file is unlinked."""
        if not components:
            raise StorageObjectError("atomic_write requires at least one component")
        _validate_components(components)
        parent_components = components[:-1]
        name = components[-1]
        with self._open_dir_chain(*parent_components) as parent_fd:
            temp_fd, temp_name = self._create_temp(parent_fd, prefix=name, mode=mode)
            try:
                self._write_fd(temp_fd, content)
                os.replace(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                os.fsync(parent_fd)
            except BaseException:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError:
                    pass
                raise
            finally:
                os.close(temp_fd)

    def atomic_publish_directory(
        self,
        *components: str,
        files: "Mapping[str, bytes]",
    ) -> None:
        """Atomically publish a new directory ``root/components`` populated
        with ``files`` (each file gets ``atomic_write`` semantics). Creates a
        uniquely-named temp directory next to the target, writes every file
        inside it, fsyncs them, then renames temp -> target and fsyncs the
        parent. On exception the temp subtree is removed."""
        if not components:
            raise StorageObjectError(
                "atomic_publish_directory requires at least one component"
            )
        _validate_components(components)
        for fname in files:
            _validate_components((fname,))
        parent_components = components[:-1]
        name = components[-1]
        with self._open_dir_chain(*parent_components) as parent_fd:
            temp_name = self._create_temp_dir(parent_fd, prefix=name)
            try:
                with self._open_under(parent_fd, temp_name) as temp_fd:
                    for fname, content in files.items():
                        file_fd = os.open(
                            fname, self._file_create_flags, 0o600, dir_fd=temp_fd
                        )
                        try:
                            self._write_fd(file_fd, content)
                        finally:
                            os.close(file_fd)
                    os.fsync(temp_fd)
                os.replace(
                    temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
                )
                os.fsync(parent_fd)
            except BaseException:
                try:
                    self._remove_tree_at(parent_fd, temp_name)
                except OSError:
                    pass
                raise

    def unlink(self, *components: str, missing_ok: bool = False) -> None:
        if not components:
            raise StorageObjectError("unlink requires at least one component")
        _validate_components(components)
        parent_components = components[:-1]
        name = components[-1]
        with self._open_dir_chain(*parent_components) as parent_fd:
            try:
                os.unlink(name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:
                if not missing_ok:
                    raise

    def remove_tree(self, *components: str) -> None:
        if not components:
            raise StorageObjectError("remove_tree requires at least one component")
        _validate_components(components)
        parent_components = components[:-1]
        name = components[-1]
        with self._open_dir_chain(*parent_components) as parent_fd:
            self._remove_tree_at(parent_fd, name)
            os.fsync(parent_fd)

    def fsync_directory(self, *components: str) -> None:
        """Fsync the directory at ``root/components`` so any prior rename /
        unlink / atomic_write landing inside it is durable. Idempotent; a
        missing directory is a no-op (the caller's prior write already failed
        or never happened)."""
        if not components:
            with self._open_root() as fd:
                os.fsync(fd)
                return
        _validate_components(components)
        try:
            with self._open_dir_chain(*components) as dir_fd:
                os.fsync(dir_fd)
        except FileNotFoundError:
            return

    # --- low-level helpers --------------------------------------------------

    def _create_temp(
        self, parent_fd: int, *, prefix: str, mode: int
    ) -> "tuple[int, str]":
        """Create a uniquely-named temp file under ``parent_fd`` and return
        (open_fd, name). The caller owns the fd."""
        for _ in range(16):
            name = f".{prefix}.{secrets.token_hex(8)}.tmp"
            try:
                fd = os.open(name, self._file_create_flags, mode, dir_fd=parent_fd)
                return fd, name
            except FileExistsError:
                continue
        raise StorageObjectError(
            f"could not create a unique temp file under {self._root}"
        )

    def _create_temp_dir(self, parent_fd: int, *, prefix: str) -> str:
        for _ in range(16):
            name = f".{prefix}.{secrets.token_hex(8)}.tmp"
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                return name
            except FileExistsError:
                continue
        raise StorageObjectError(
            f"could not create a unique temp directory under {self._root}"
        )

    def _write_fd(self, fd: int, content: bytes) -> None:
        written = 0
        while written < len(content):
            n = os.write(fd, content[written:])
            if n == 0:
                raise StorageObjectError("write returned 0 bytes")
            written += n
        os.fsync(fd)

    @contextmanager
    def _open_under(self, parent_fd: int, name: str) -> "Iterator[int]":
        fd = os.open(name, self._dir_flags, dir_fd=parent_fd)
        try:
            yield fd
        finally:
            os.close(fd)

    def _remove_tree_at(self, parent_fd: int, name: str) -> None:
        """Remove the file or directory ``name`` living under ``parent_fd``,
        recursively. Refuses to follow a symlink: a symlink is unlinked, not
        traversed."""
        try:
            sub_fd = os.open(name, self._dir_flags, dir_fd=parent_fd)
        except NotADirectoryError:
            # Not a directory: unlink it directly (O_NOFOLLOW guarantees we
            # are NOT following a symlink to do so).
            os.unlink(name, dir_fd=parent_fd)
            return
        except FileNotFoundError:
            return
        try:
            for entry in os.listdir(sub_fd):
                self._remove_tree_at(sub_fd, entry)
        finally:
            os.close(sub_fd)
        os.rmdir(name, dir_fd=parent_fd)


class DirectoryIO:
    """Common directory I/O contract used by both platform modes."""


class PosixSecureDirectory(SecureDirectory, DirectoryIO):
    pass


class TrustedLocalDirectory(DirectoryIO):
    """Path-based I/O for an explicitly trusted, single-user local root.

    This mode intentionally makes no symlink or TOCTOU guarantee and must not
    be used for multi-tenant data. It is separate from the POSIX implementation
    so importing and constructing it never evaluates POSIX-only flags.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._mode = FilesystemSecurityMode.TRUSTED_LOCAL

    @property
    def root(self) -> Path: return self._root
    @property
    def security_mode(self) -> FilesystemSecurityMode: return self._mode

    def _path(self, parts: tuple[str, ...]) -> Path:
        _validate_components(parts)
        return self._root.joinpath(*parts)

    def ensure_directory(self, *parts: str, mode: int = 0o700) -> None:
        self._path(parts).mkdir(parents=True, exist_ok=True, mode=mode)
    def read_bytes(self, *parts: str) -> bytes: return self._path(parts).read_bytes()
    def stat(self, *parts: str):
        try: return self._path(parts).lstat()
        except FileNotFoundError: return None
    def list_names(self, *parts: str) -> tuple[str, ...]:
        path = self._path(parts)
        return tuple(sorted(p.name for p in path.iterdir())) if path.exists() else ()
    def atomic_write(self, *parts: str, content: bytes, mode: int = 0o600) -> None:
        path = self._path(parts); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(content); tmp.replace(path)
    def atomic_publish_directory(self, *parts: str, files: Mapping[str, bytes]) -> None:
        path = self._path(parts); tmp = path.with_name(f".{path.name}.tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        for name, content in files.items(): (tmp / name).write_bytes(content)
        tmp.replace(path)
    def unlink(self, *parts: str, missing_ok: bool = False) -> None: self._path(parts).unlink(missing_ok=missing_ok)
    def remove_tree(self, *parts: str) -> None:
        import shutil
        shutil.rmtree(self._path(parts), ignore_errors=True)
    def fsync_directory(self, *parts: str) -> None:
        import os as _os
        path = self._path(parts)
        if not path.exists():
            return
        fd = _os.open(str(path), _os.O_RDONLY)
        try:
            _os.fsync(fd)
        finally:
            _os.close(fd)
