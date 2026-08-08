#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional Git support backed by Dulwich on Python 3.10+."""

from linktools.errors import GitError, GitDivergedError, GitUnavailableError, missing_optional_class

from ._support import (
    get_git_unavailable_reason,
    is_git_available,
    require_git_available,
)
from ._sync import GitSyncPolicy
from ._progress import GitProgressStream

try:
    require_git_available()
except GitUnavailableError as exc:
    GitRepository = missing_optional_class("GitRepository", "git", exc)
    GitHead = missing_optional_class("GitHead", "git", exc)
else:
    from ._repository import GitHead, GitRepository

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
