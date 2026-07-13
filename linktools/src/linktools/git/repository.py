#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Git repository wrapper backed by dulwich.

Pure-Python -- never shells out to system ``git``. Clone is atomic (clone to a
staging dir, then rename) so an interrupted clone never leaves a half-valid
repository. Write operations are serialised per repository via the
environment's LockManager.
"""

import contextlib
import os
import shutil
import uuid
from typing import TYPE_CHECKING

from dulwich import porcelain
from dulwich.errors import GitProtocolError
from dulwich.repo import Repo as DulwichRepo

from linktools import utils
from linktools.core import environ
from linktools.errors import GitError, GitDivergedError, GitStashRestoreError
from linktools.rich import create_progress

from .progress import GitProgressStream
from .sync import GitSyncPolicy

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType

_logger = environ.get_logger("git")


@contextlib.contextmanager
def _wrap_protocol_errors():
    """Turn a transport/protocol failure (bad URL, auth rejected, server
    error, ...) into a plain ``GitError`` -- callers must never need to
    import dulwich just to catch its own transport exception type."""
    try:
        yield
    except GitProtocolError as exc:
        raise GitError(str(exc)) from exc


class GitHead(object):
    """A local git branch that can be checked out."""

    def __init__(self, repository: "GitRepository", name: str) -> None:
        self._repository = repository
        self.name = name

    def checkout(self):
        """Check out this branch in the working tree, serialized through
        the owning repository's write lock -- same as every other write
        operation on this repository."""
        with self._repository._write_lock():
            self._repository._checkout(self.name)


class GitRepository(object):
    """Pure-Python git repository wrapper backed by dulwich."""

    def __init__(self, environ: "Any", path: "PathType") -> None:
        self._environ = environ
        self._path = str(path)
        self._repo = DulwichRepo(self._path)  # raises NotGitRepository if invalid

    @classmethod
    def open_if_valid(cls, environ: "Any", path: "PathType") -> "GitRepository | None":
        """Like the constructor, but returns ``None`` instead of raising when
        ``path`` isn't usable as a git repository -- not a repository at all
        (dulwich's ``NotGitRepository``), or one whose on-disk state dulwich
        can't even open (a corrupted object store/index, a permission
        error, ...). dulwich's exception types have no single common base
        class to catch narrowly, so this catches broadly by design: the
        whole point of this method is turning "can't open this as a repo,
        for any reason" into ``None`` rather than requiring every caller to
        import dulwich just to catch its exception types itself."""
        try:
            return cls(environ, path)
        except Exception:
            return None

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
        return bool(any(status.staged.values()) or status.unstaged or status.untracked)

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
        with self._write_lock():
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
        with self._write_lock(), _wrap_protocol_errors():
            refspecs = branch and self._branch_ref(branch).decode()
            porcelain.push(self._path, remote_location, refspecs, force=force)

    def create_head(self, branch: str) -> "GitHead":
        with self._write_lock():
            return self._create_head_unlocked(branch)

    def _create_head_unlocked(self, branch: str) -> "GitHead":
        # Caller must already hold self._write_lock() -- a process_lock()
        # is a fresh OS-level file lock every call, not a reentrant one, so
        # acquiring it again here (from checkout_or_create(), which also
        # needs this) would deadlock rather than no-op.
        branch_ref = self._branch_ref(branch)
        target = self._remote_branch_target(branch)
        if target is None:
            with _wrap_protocol_errors():
                result = porcelain.fetch(self._path, depth=1, force=True, quiet=True)
            target = result.refs.get(branch_ref)
        if target is None:
            raise GitError("Remote branch `%s` not found." % branch)
        porcelain.branch_create(self._path, branch, target)
        return GitHead(self, branch)

    def checkout_or_create(self, branch: str) -> None:
        """Check out ``branch``, creating it from the remote if it doesn't
        exist locally -- both branches (existing or newly created) run
        under a single write-lock acquisition, not two nested ones."""
        with self._write_lock():
            if branch in self.heads:
                self._checkout(branch)
            else:
                self._create_head_unlocked(branch)
                self._checkout(branch)

    # -- sync ( ------------------------------------------------------

    def sync(self, policy: str = GitSyncPolicy.FAST_FORWARD_ONLY) -> None:
        """Reconcile the current branch with its remote per to ``policy``."""
        with self._write_lock():
            if policy == GitSyncPolicy.FAIL_IF_DIRTY and self.is_dirty():
                raise GitError("Working tree is dirty; refusing to sync.")
            stashed = False
            if policy == GitSyncPolicy.STASH_AND_RESTORE and self.is_dirty():
                self._stash_push()
                stashed = True

            sync_exc = None
            try:
                if policy == GitSyncPolicy.RESET_TO_REMOTE:
                    self._force_update()
                else:
                    self._fast_forward()  # FAST_FORWARD_ONLY / default
            except BaseException as exc:
                sync_exc = exc

            if stashed:
                try:
                    self._stash_pop()
                except Exception as restore_exc:
                    # A restore failure must never silently replace the
                    # original sync failure (a plain try/finally would do
                    # exactly that) -- combine both into one message,
                    # chained from whichever is the more relevant cause.
                    if sync_exc is not None:
                        raise GitStashRestoreError(
                            "sync failed (%s: %s) and restoring the stashed "
                            "changes afterward also failed (%s: %s)"
                            % (type(sync_exc).__name__, sync_exc,
                               type(restore_exc).__name__, restore_exc)
                        ) from sync_exc
                    raise GitStashRestoreError(
                        "sync succeeded but restoring the stashed changes "
                        "afterward failed: %s: %s"
                        % (type(restore_exc).__name__, restore_exc)
                    ) from restore_exc

            if sync_exc is not None:
                raise sync_exc

    # -- low-level porcelain wrappers ---------------------------------------

    def _stash_push(self):
        porcelain.stash_push(self._path)

    def _stash_pop(self):
        porcelain.stash_pop(self._path)

    def _reset_hard(self):
        porcelain.reset(self._path, mode="hard")

    def _checkout(self, branch):
        porcelain.switch(self._path, branch)

    def _fast_forward(self):
        branch_ref = self._current_branch_ref()
        with create_progress("message") as progress:
            try:
                with _wrap_protocol_errors():
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
        with create_progress("message") as progress, _wrap_protocol_errors():
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
        """Shallow-clone, atomically: clone to staging -> rename.

        An interrupted clone leaves only the staging dir behind, never a
        half-valid repository at ``repo_path``.
        """
        target = str(repo_path)
        # lexists, not exists -- a dangling symlink already at the target
        # path must also be rejected: os.path.exists() follows it and
        # (since the target is missing) reports False, which would let
        # porcelain.clone() try to write through/replace it instead of
        # this classmethod ever seeing it as an occupied path.
        if os.path.lexists(target):
            raise GitError("Clone target already exists: %s" % target)
        staging = "%s.staging-%s" % (target, uuid.uuid4().hex[:8])
        kwargs = {}
        if branch:
            kwargs["branch"] = branch
        try:
            with create_progress("message") as progress, _wrap_protocol_errors():
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
