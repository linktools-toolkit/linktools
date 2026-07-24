#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filesystem path-safety primitives for :class:`~linktools.ai.asset.file.FileAssetBackend`
(covering its data/metadata/whiteouts/idempotency subdirectories uniformly): a
lexical, per-component ``lstat()`` walk from a trusted root down to a
candidate path, so a symlink planted anywhere in the chain -- the target
itself, or an ancestor directory -- is caught before the backend reads or
writes through it.

:class:`SymlinkPolicy` governs what happens when a component IS a symlink:

- ``DENY`` -- reject unconditionally, wherever it appears in the chain.
- ``ALLOW_INTERNAL`` -- allow it, but only if resolving it never leaves the
  trusted root (a symlink that redirects elsewhere inside the root is a
  legitimate internal indirection; one that escapes it is not).

A component that does not yet exist ends the walk (nothing deeper can exist
either), so a create/write targeting a new file is not rejected merely for
not existing yet -- only the nearest EXISTING ancestor must be safe."""

import os
import secrets
import stat as _stat
from enum import Enum
from pathlib import Path

from ..errors import InvalidAssetPathError


class SymlinkPolicy(str, Enum):
    DENY = "deny"
    ALLOW_INTERNAL = "allow_internal"


def _validate_components(components: "tuple[str, ...]") -> None:
    for part in components:
        if not part:
            raise InvalidAssetPathError("empty path component not allowed")
        if part in (".", ".."):
            raise InvalidAssetPathError(f"path traversal not allowed: {part!r}")
        if "\x00" in part:
            raise InvalidAssetPathError("NUL byte not allowed in path component")


def resolve_secure_path(root: Path, *components: str, policy: SymlinkPolicy) -> Path:
    """Lexically construct ``root / components...`` and walk it component by
    component, ``lstat()``-ing each one that EXISTS. Returns the constructed
    (not OS-resolved) path so a legitimate ``ALLOW_INTERNAL`` symlink is still
    followed naturally by the eventual file operation; this function only
    decides whether the chain is SAFE to use, not what path to substitute."""
    _validate_components(components)
    root_resolved = root.resolve()
    current = root
    for part in components:
        current = current / part
        try:
            st = os.lstat(current)
        except OSError:
            # This component -- and therefore every deeper one -- does not
            # exist yet. A create/write targeting a non-existent tail is
            # allowed; only existing ancestors must be checked.
            break
        if _stat.S_ISLNK(st.st_mode):
            if policy is SymlinkPolicy.DENY:
                raise InvalidAssetPathError(f"symlink not allowed: {current}")
            resolved = current.resolve()
            if resolved != root_resolved and root_resolved not in resolved.parents:
                raise InvalidAssetPathError(
                    f"symlink escapes root: {current} -> {resolved}"
                )
    return root / Path(*components)


def open_temp_nofollow(
    directory: Path, *, prefix: str = ".", suffix: str = ".tmp"
) -> "tuple[int, Path]":
    """Create a uniquely-named temp file in ``directory`` with
    ``O_CREAT|O_EXCL|O_NOFOLLOW`` (plus ``O_CLOEXEC`` where available) and
    return its open fd + path. ``O_EXCL`` already refuses an existing name
    (symlink or not); ``O_NOFOLLOW`` is defense in depth against the
    vanishingly unlikely case of a colliding pre-planted symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(10):
        name = f"{prefix}{secrets.token_hex(8)}{suffix}"
        path = directory / name
        try:
            fd = os.open(str(path), flags, 0o600)
            return fd, path
        except FileExistsError:
            continue
    raise OSError(f"could not create a unique temp file under {directory}")


__all__: "list[str]" = ["SymlinkPolicy", "resolve_secure_path", "open_temp_nofollow"]
