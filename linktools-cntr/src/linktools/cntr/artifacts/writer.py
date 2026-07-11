#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic writer for generated files (Spec section 29): same-directory
temp file, flush, fsync, ``os.replace`` -- reusing ``linktools.utils.
atomic_write`` for that mechanics -- plus a same-content short-circuit so an
unchanged generated file keeps its original mtime and no write happens at
all.
"""
import os
import stat
from typing import TYPE_CHECKING

from linktools import utils

if TYPE_CHECKING:
    from linktools.types import PathType


def atomic_write_text_if_changed(path: "PathType", content: str, encoding: str = "utf-8") -> bool:
    """Write ``content`` to ``path`` atomically. Return True iff it changed.

    ``linktools.utils.atomic_write`` replaces the target with a freshly
    created temp file (``tempfile.mkstemp``, mode 0600), which would
    otherwise silently narrow an existing file's permissions on every
    regeneration; the previous mode is restored here for an existing target.
    """
    path = str(path)
    original_mode = None
    if os.path.exists(path):
        with open(path, "r", encoding=encoding) as f:
            existing = f.read()
        if existing == content:
            return False
        original_mode = stat.S_IMODE(os.stat(path).st_mode)
    utils.atomic_write(path, content, encoding=encoding)
    if original_mode is not None:
        os.chmod(path, original_mode)
    return True
