#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
    from ..types import PathType


def get_hash(data: "str | bytes", algorithm: "Literal['md5', 'sha1', 'sha256']" = "md5") -> str:
    """Return the digest for bytes or text using the selected hash algorithm."""
    import hashlib

    if isinstance(data, str):
        data = bytes(data, "utf8")
    m = getattr(hashlib, algorithm)()
    m.update(data)
    return m.hexdigest()


def get_file_hash(path: "PathType", algorithm: "Literal['md5', 'sha1', 'sha256']" = "md5") -> str:
    """Return the digest for a file using the selected hash algorithm."""
    import hashlib

    m = getattr(hashlib, algorithm)()
    with open(path, "rb") as fd:
        while True:
            data = fd.read(4096 << 4)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def get_md5(data: "str | bytes") -> str:
    """Return the MD5 digest for bytes or text."""
    return get_hash(data, algorithm="md5")


def get_file_md5(path: "PathType"):
    """Return the MD5 digest for a file."""
    return get_file_hash(path, algorithm="md5")


def get_hash_ident(data: "str | bytes"):
    """Return a short stable identifier from a hashed value."""
    if isinstance(data, str):
        data = bytes(data, "utf8")
    length = f"{len(data):0>4x}"
    md5 = get_hash(data, "md5")
    sha1 = get_hash(data, "sha1")
    return f"{length[-4:]}{md5[:6]}{sha1[:6]}"
