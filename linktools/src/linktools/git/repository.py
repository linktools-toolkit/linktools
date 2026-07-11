#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Git repository wrapper backed by dulwich (spec §12).

Pure-Python (dulwich) -- never shells out to system ``git`` (spec §12.1). The
public surface (``GitRepository``/``GitHead``) is stable; cntr depends on it.
New behaviour vs the legacy single file:

* atomic clone (§12.2): clone to a staging dir, then rename, so an interrupted
  clone never leaves a half-valid repository;
* ``sync(policy=)`` (§12.3) replaces ``pull(reset=...)`` with an explicit
  :class:`~linktools.git.sync.GitSyncPolicy`;
* write operations are serialised per repository via the environment's
  LockManager (§12.8 GIT-006).
"""

import contextlib
import os
import shutil
import uuid
from typing import TYPE_CHECKING

from dulwich import porcelain
from dulwich.repo import Repo as DulwichRepo

from linktools import utils
from linktools.core import environ
from linktools.errors import GitError, GitDivergedError
from linktools.rich import create_progress

from .progress import GitProgressStream
from .sync import GitSyncPolicy

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType

_logger = environ.get_logger("git")


class _GitProxy(object):
    """Common git operations on a repo path via dulwich porcelain."""

    def __init__(self, path: str) -> None:
        self._path = path

    def stash(self, *args):
        if args and args[0] == "pop":
            porcelain.stash_pop(self._path)
        else:
            porcelain.stash_push(self._path)

    def reset(self, hard=False, **kwargs):
        porcelain.reset(self._path, mode="hard" if hard else "mixed")

    def checkout(self, branch):
        porcelain.switch(self._path, branch)


class GitHead(object):
    """A local git branch that can be checked out."""

    def __init__(self, path: str, name: str) -> None:
        self._path = path
        self.name = name

    def checkout(self):
        """Check out this branch in the working tree."""
        porcelain.switch(self._path, self.name)


class GitRepository(object):
    """Pure-Python git repository wrapper backed by dulwich."""

    def __init__(self, environ: "Any", path: "PathType") -> None:
        self._environ = environ
        self._path = str(path)
        self._repo = DulwichRepo(self._path)  # raises NotGitRepository if invalid
        self.git = _GitProxy(self._path)

    def close(self):
        """Release the underlying repository handle (open pack files, etc.)."""
        self._repo.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    # -- locking ( ---------------------------------------------------

    @contextlib.contextmanager
    def _write_lock(self):
        # Serialise write operations on this repository path (uses the
        # injected environ's LockManager, not the module-level global).
        key = "git-repo:" + utils.get_hash(self._path, "sha256")
        with self._environ.locks.process_lock(key):
            yield

    # -- reads -------------------------------------------------------------

    @property
    def heads(self):
        refs = self._repo.refs.as_dict(b"refs/heads/")
        return [name.decode() for name in refs]

    def status(self):
        return porcelain.status(self._path)

    def is_dirty(self):
        status = self.status()
        return bool(any(status.staged.values()) or status.unstaged)

    def head_sha(self) -> str:
        """Current HEAD commit SHA (full hex string)."""
        return self._repo.head().decode()

    def current_branch(self) -> "str | None":
        """Current branch name, or None if HEAD is detached."""
        try:
            branch_ref = self._current_branch_ref()
        except GitError:
            return None
        return branch_ref[len(b"refs/heads/"):].decode()

    # -- writes (all serialised) ------------------------------------------

    def add(self, *paths):
        porcelain.add(self._path, list(paths) or None)

    def commit(self, message: str, author: str = None, committer: str = None,
               all: bool = False) -> str:
        with self._write_lock():
            sha = porcelain.commit(
                self._path,
                message=message,
                author=author.encode() if author else None,
                committer=committer.encode() if committer else None,
                all=all,
            )
            return sha.decode()

    def push(self, remote_location=None, branch=None, force=False):
        with self._write_lock():
            refspecs = branch and self._branch_ref(branch).decode()
            porcelain.push(self._path, remote_location, refspecs, force=force)

    def create_head(self, branch: str) -> "GitHead":
        with self._write_lock():
            branch_ref = self._branch_ref(branch)
            target = self._remote_branch_target(branch)
            if target is None:
                result = porcelain.fetch(self._path, depth=1, force=True, quiet=True)
                target = result.refs.get(branch_ref)
            if target is None:
                raise GitError("Remote branch `%s` not found." % branch)
            porcelain.branch_create(self._path, branch, target)
            return GitHead(self._path, branch)

    # -- sync ( ------------------------------------------------------

    def sync(self, policy: str = GitSyncPolicy.FAST_FORWARD_ONLY) -> None:
        """Reconcile the current branch with its remote per to ``policy``."""
        with self._write_lock():
            if policy == GitSyncPolicy.FAIL_IF_DIRTY and self.is_dirty():
                raise GitError("Working tree is dirty; refusing to sync.")
            stashed = False
            if policy == GitSyncPolicy.STASH_AND_RESTORE and self.is_dirty():
                self.git.stash()
                stashed = True
            try:
                if policy == GitSyncPolicy.RESET_TO_REMOTE:
                    self._force_update()
                else:
                    self._fast_forward()  # FAST_FORWARD_ONLY / default
            finally:
                if stashed:
                    self.git.stash("pop")

    def pull(self, reset: bool = False) -> None:
        """Legacy entry point; maps to :meth:`sync` with the equivalent policy."""
        self.sync(policy=GitSyncPolicy.RESET_TO_REMOTE if reset
                  else GitSyncPolicy.FAST_FORWARD_ONLY)

    def _fast_forward(self):
        branch_ref = self._current_branch_ref()
        with create_progress("message") as progress:
            try:
                porcelain.pull(
                    self._path,
                    refspecs=branch_ref,
                    errstream=GitProgressStream(progress),
                )
            except porcelain.DivergedBranches:
                raise GitDivergedError(
                    "Local branch has diverged from the remote and cannot be fast-forwarded."
                )

    def _force_update(self):
        # These repos are shallow (depth=1) clones, so dulwich cannot merge or
        # rebase a diverged branch. Fetch remote objects and hard-reset.
        branch_ref = self._current_branch_ref()
        with create_progress("message") as progress:
            result = porcelain.fetch(
                self._path,
                errstream=GitProgressStream(progress),
                depth=1,
                force=True,
            )
        target = result.refs.get(branch_ref)
        if target is None:
            raise GitError("Remote branch `%s` not found." % branch_ref.decode())
        porcelain.reset(self._path, "hard", target)

    # -- ref helpers -------------------------------------------------------

    def _branch_ref(self, branch: str) -> bytes:
        return b"refs/heads/" + branch.encode()

    def _current_branch_ref(self):
        head_refs, _ = self._repo.refs.follow(b"HEAD")
        branch_ref = head_refs[-1]
        if not branch_ref.startswith(b"refs/heads/"):
            raise GitError("Repository HEAD is detached; unable to resolve branch to update.")
        return branch_ref

    def _remote_branch_target(self, branch):
        remote_ref = b"refs/remotes/origin/" + branch.encode()
        return self._repo.refs.as_dict().get(remote_ref)

    # -- clone ( atomic) ---------------------------------------------

    @classmethod
    def clone(cls, environ: "Any", url: str, repo_path: str = None,
              branch: str = None) -> "GitRepository":
        """Shallow-clone, atomically: clone to staging -> rename (spec §12.2).

        An interrupted clone leaves only the staging dir behind, never a
        half-valid repository at ``repo_path``.
        """
        target = str(repo_path)
        if os.path.exists(target):
            raise GitError("Clone target already exists: %s" % target)
        staging = "%s.staging-%s" % (target, uuid.uuid4().hex[:8])
        kwargs = {}
        if branch:
            kwargs["branch"] = branch
        try:
            with create_progress("message") as progress:
                porcelain.clone(
                    url,
                    staging,
                    depth=1,
                    errstream=GitProgressStream(progress),
                    **kwargs
                ).close()
            os.replace(staging, target)  # atomic on the same filesystem
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls(environ, target)
