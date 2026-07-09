#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
from pathlib import Path
from typing import overload, TYPE_CHECKING

from ..errors import Error
from ._common import get_environ, get_logger

DEFAULT_ENCODING = "utf-8"


def is_sub_path(path, root_path) -> bool:
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
        abs_root_path = os.path.abspath(os.path.expanduser(root_path))
        return os.path.commonpath([abs_path, abs_root_path]) == abs_root_path
    except ValueError:
        return False


def join_path(root_path, *paths: str) -> "Path":
    target_path = str(root_path)
    for path in paths:
        parent_path = target_path
        target_path = os.path.abspath(os.path.join(target_path, path))
        try:
            if os.path.commonpath([target_path, parent_path]) != parent_path:
                raise Error(f"Unsafe path \"{path}\"")
        except ValueError:
            raise Error(f"Unsafe path \"{path}\"")
    return Path(target_path)


def ensure_within(path, root):
    """Raise Error if ``path`` does not resolve within ``root`` (spec §17.2).

    Unlike ``is_sub_path`` (which returns a bool), this raises on violation and
    returns the resolved absolute Path on success.
    """
    if not is_sub_path(str(path), str(root)):
        raise Error("path %r is not within root %r" % (str(path), str(root)))
    return Path(os.path.abspath(str(path)))


# Alias for the spec's preferred name (
safe_join = join_path


if TYPE_CHECKING:

    @overload
    def read_file(path):
        ...


    @overload
    def read_file(path, text: False):
        ...


    @overload
    def read_file(path, text: True, encoding: str = DEFAULT_ENCODING):
        ...


    @overload
    def read_file(path, text: bool, encoding: str = DEFAULT_ENCODING):
        ...


def read_file(path, text: bool = False, encoding: str = DEFAULT_ENCODING):
    if text:
        with open(path, "rt", encoding=encoding) as fd:
            return fd.read()
    with open(path, "rb") as fd:
        return fd.read()


def write_file(path, data, encoding: str = DEFAULT_ENCODING) -> None:
    if isinstance(data, str):
        with open(path, "wt", encoding=encoding) as fd:
            fd.write(data)
    else:
        with open(path, "wb") as fd:
            fd.write(data)


def atomic_write(path, data, encoding: str = DEFAULT_ENCODING) -> None:
    """Write ``data`` to ``path`` atomically (spec §3.7/§17.1 UTL-001).

    temp-in-same-dir -> write -> flush -> fsync -> os.replace. A crash mid-write
    leaves the previous version (if any) intact and no partial file at ``path``.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            if isinstance(data, str):
                handle.write(data.encode(encoding))
            else:
                handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, str(target))
    except BaseException:
        # Never leave the .tmp behind on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_replace(source, target) -> None:
    """Atomically replace ``target`` with ``source`` (spec §17.1 UTL-001)."""
    os.replace(str(source), str(target))


def remove_file(path) -> None:
    if not os.path.exists(path):
        return
    environ = get_environ()
    if environ.debug:
        get_logger().debug(f"Remove File: {path}")
    if os.path.isdir(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            os.remove(path)
        except Exception:
            pass


def clear_directory(path) -> None:
    if not os.path.isdir(path):
        return
    if get_environ().debug:
        get_logger().debug(f"Clear Directory: {path}")
    for name in os.listdir(path):
        target_path = os.path.join(path, name)
        if os.path.isdir(target_path):
            import shutil
            shutil.rmtree(target_path, ignore_errors=True)
        else:
            try:
                os.remove(target_path)
            except Exception:
                pass


def safe_remove(path, root=None) -> bool:
    """Remove a file or directory after verifying it is within ``root`` (§17.5).

    ``root`` defaults to the parent directory of ``path``. A target resolving
    outside ``root`` raises :class:`linktools.errors.Error` and is left
    untouched. Returns True if something was removed, False if absent.
    """
    import shutil

    target = Path(str(path))
    boundary = Path(str(root)) if root is not None else target.parent
    if not is_sub_path(str(target), str(boundary)):
        raise Error("Refusing to remove %r: outside root %r" % (str(target), str(boundary)))
    if not target.exists() and not target.is_symlink():
        return False
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(str(target), ignore_errors=True)
    else:
        try:
            target.unlink()
        except FileNotFoundError:
            return False
    return True


def safe_rmtree(path, root=None) -> bool:
    """Remove a directory tree after verifying it is within ``root`` (§17.5)."""
    return safe_remove(path, root=root)
