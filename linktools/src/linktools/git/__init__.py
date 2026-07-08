#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure-Python (dulwich) git support (spec §12).

Public API (stable -- cntr depends on it)::

    from linktools.git import GitRepository, GitHead, GitSyncPolicy
    from linktools.git import GitError, GitDivergedError

``import linktools.git`` does NOT require dulwich: only ``GitRepository`` /
``GitHead`` do (they need the optional ``linktools[git]`` extra). ``GitSyncPolicy``
and ``GitProgressStream`` are dulwich-free and always importable. Without dulwich
the repository classes become placeholders that raise ``ModuleError`` on use.
"""

from linktools.errors import GitError, GitDivergedError, missing_optional_class

__all__ = [
    "GitRepository", "GitHead", "GitSyncPolicy", "GitProgressStream",
    "GitError", "GitDivergedError",
]


# dulwich-free modules -- always importable.
from .sync import GitSyncPolicy
from .progress import GitProgressStream

try:
    from .repository import GitRepository, GitHead
except ImportError as _exc:  # dulwich not installed
    # Only swallow a missing dulwich; re-raise internal ImportErrors so a real
    # bug in repository.py is not masked as "optional dependency absent".
    _missing = (getattr(_exc, "name", "") or "").split(".", 1)[0]
    if _missing != "dulwich":
        raise
    GitRepository = missing_optional_class("GitRepository", "git", _exc)
    GitHead = missing_optional_class("GitHead", "git", _exc)
