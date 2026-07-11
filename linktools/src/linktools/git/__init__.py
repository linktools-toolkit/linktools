#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional Git support backed by Dulwich on Python 3.10+."""

from linktools.errors import GitError, GitDivergedError, GitUnavailableError, missing_optional_class

from .support import (
    get_git_unavailable_reason,
    is_git_available,
    require_git_available,
)
from .sync import GitSyncPolicy
from .progress import GitProgressStream

try:
    require_git_available()
except GitUnavailableError as exc:
    GitRepository = missing_optional_class("GitRepository", "git", exc)
    GitHead = missing_optional_class("GitHead", "git", exc)
else:
    from .repository import GitHead, GitRepository

__all__ = [
    "GitRepository",
    "GitHead",
    "GitSyncPolicy",
    "GitProgressStream",
    "GitError",
    "GitDivergedError",
    "GitUnavailableError",
    "is_git_available",
    "get_git_unavailable_reason",
    "require_git_available",
]
