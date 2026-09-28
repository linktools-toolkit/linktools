#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Git sync policies."""


class GitSyncPolicy(object):
    """How a local branch is reconciled with its remote.

    * ``FAST_FORWARD_ONLY`` -- update only if the remote is a fast-forward;
      raise :class:`linktools.errors.GitDivergedError` otherwise.
    * ``RESET_TO_REMOTE`` -- hard-reset the local branch to the remote (the
      recovery path for diverged shallow clones, which dulwich cannot merge).
    * ``FAIL_IF_DIRTY`` -- refuse to touch a dirty working tree.
    * ``STASH_AND_RESTORE`` -- stash local changes, sync, then restore.
    """

    FAST_FORWARD_ONLY = "fast_forward_only"
    RESET_TO_REMOTE = "reset_to_remote"
    FAIL_IF_DIRTY = "fail_if_dirty"
    STASH_AND_RESTORE = "stash_and_restore"

    ALL = (FAST_FORWARD_ONLY, RESET_TO_REMOTE, FAIL_IF_DIRTY, STASH_AND_RESTORE)
