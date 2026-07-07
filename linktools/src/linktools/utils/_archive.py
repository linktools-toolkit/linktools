#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Safe archive extraction (spec §10.7 TOL-007).

Extracts zip/tar archives while refusing anything that could escape the
destination or exhaust resources:

* absolute paths, ``..`` traversal, drive letters;
* symlinks / hardlinks / device / fifo files (by default);
* too many entries, oversized single files, oversized total payload.

Every extracted path is verified to resolve inside the destination root.
"""

import os
import tarfile
import zipfile
from typing import Any, Optional

from ..errors import ToolArchiveError

__all__ = ["safe_extract"]

_DEFAULT_MAX_FILES = 100000
_DEFAULT_MAX_TOTAL = 10 * 1024 * 1024 * 1024  # 10 GiB
_DEFAULT_MAX_FILE = 2 * 1024 * 1024 * 1024     # 2 GiB


def _within(path, root):
    # type: (str, str) -> bool
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _check_member_name(name, dest):
    # type: (str, str) -> str
    """Reject dangerous entry names and return the resolved target path."""
    # Normalize separators and reject absolute / drive / traversal.
    norm = name.replace("\\", "/")
    if norm.startswith("/"):
        raise ToolArchiveError("refusing absolute archive path: %r" % name)
    # Windows drive letter (e.g. C:\...) -- detect before normalisation too.
    if len(name) >= 2 and name[1] == ":":
        raise ToolArchiveError("refusing drive-letter archive path: %r" % name)
    target = os.path.join(dest, norm)
    target = os.path.normpath(target)
    if not _within(target, dest):
        raise ToolArchiveError("archive path escapes destination: %r" % name)
    # Re-check for ".." components after normalisation.
    if os.pardir in target.split(os.sep)[len(os.path.normpath(dest).split(os.sep)):]:
        raise ToolArchiveError("archive path contains traversal: %r" % name)
    return target


def _detect(path):
    # type: (str) -> str
    with open(path, "rb") as handle:
        magic = handle.read(4)
    if magic.startswith(b"PK\x03\x04") or magic.startswith(b"PK\x05\x06"):
        return "zip"
    if magic[:2] in (b"\x1f\x8b",) or (len(magic) >= 3 and magic[0] == 0x75 and magic[1] == 0x73 and magic[2] == 0x74):
        return "tar"  # gzip or uncompressed tar
    if zipfile.is_zipfile(path):
        return "zip"
    if tarfile.is_tarfile(path):
        return "tar"
    raise ToolArchiveError("unrecognised archive format: %s" % path)


def _extract_zip(path, dest, max_files, max_total, max_file):
    seen = set()  # type: set
    total = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if name.endswith("/"):
                continue  # directory entry
            _reject_zip_symlink(info, name)
            if len(seen) >= max_files:
                raise ToolArchiveError("archive exceeds max_files=%d" % max_files)
            if info.file_size > max_file:
                raise ToolArchiveError("archive entry %r exceeds max_file_size" % name)
            total += info.file_size
            if total > max_total:
                raise ToolArchiveError("archive exceeds max_total_size")
            target = _check_member_name(name, dest)
            if target in seen:
                raise ToolArchiveError("archive contains duplicate path: %r" % name)
            seen.add(target)
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                while True:
                    chunk = src.read(64 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)


def _reject_zip_symlink(info, name):
    # unix mode bits live in the top 16 bits of external_attr.
    mode = (info.external_attr >> 16) & 0o170000
    if mode in (0o120000, 0o140000):  # symlink, socket
        raise ToolArchiveError("refusing archive link entry: %r" % name)
    if mode in (0o020000, 0o060000, 0o010000, 0o040000):  # char/block dev, fifo
        raise ToolArchiveError("refusing archive device/fifo entry: %r" % name)


def _extract_tar(path, dest, max_files, max_total, max_file):
    seen = set()  # type: set
    total = 0
    with tarfile.open(path) as t:
        for member in t.getmembers():
            if member.issym() or member.islnk():
                raise ToolArchiveError("refusing archive link entry: %r" % member.name)
            if member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                raise ToolArchiveError("refusing archive device/fifo entry: %r" % member.name)
            if not member.isfile():
                continue  # directories handled on demand
            name = member.name
            if len(seen) >= max_files:
                raise ToolArchiveError("archive exceeds max_files=%d" % max_files)
            if member.size > max_file:
                raise ToolArchiveError("archive entry %r exceeds max_file_size" % name)
            total += member.size
            if total > max_total:
                raise ToolArchiveError("archive exceeds max_total_size")
            target = _check_member_name(name, dest)
            if target in seen:
                raise ToolArchiveError("archive contains duplicate path: %r" % name)
            seen.add(target)
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            src = t.extractfile(member)
            try:
                with open(target, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            finally:
                if src is not None:
                    src.close()


def safe_extract(archive_path, dest, *, max_files=_DEFAULT_MAX_FILES,
                 max_total_size=_DEFAULT_MAX_TOTAL, max_file_size=_DEFAULT_MAX_FILE):
    # type: (Any, Any, int, int, int) -> None
    """Extract ``archive_path`` (zip or tar) into ``dest`` safely (§10.7).

    v4 §9.10: ``dest`` must be empty (no existing files). This prevents
    accidental overwrite of existing tool installations.
    Refuses path traversal, absolute/drive paths, symlink/hardlink/device/fifo
    entries, and enforces file-count / per-file / total-size caps.
    """
    archive_path = str(archive_path)
    dest = str(dest)
    os.makedirs(dest, exist_ok=True)
    # v4 §9.10: dest must be empty.
    if os.path.isdir(dest) and os.listdir(dest):
        from ..errors import ToolArchiveError
        raise ToolArchiveError(
            "safe_extract destination is not empty: %s" % dest)
    kind = _detect(archive_path)
    if kind == "zip":
        _extract_zip(archive_path, dest, max_files, max_total_size, max_file_size)
    else:
        _extract_tar(archive_path, dest, max_files, max_total_size, max_file_size)
