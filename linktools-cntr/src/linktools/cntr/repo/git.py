#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional Git operations for repositories: capability gating, clone,
update (clone-on-demand / branch checkout / fast-forward or force sync) and
read-only revision/dirty inspection.

Never imports dulwich directly -- all Dulwich-backed behaviour is reached
through ``linktools.git``'s public API. A missing/unsupported Git runtime
degrades to an explicit warning plus failure/`unsupported` marker; local
(non-git) repositories never need this module at all.
"""
import os
from collections import namedtuple
from typing import TYPE_CHECKING

from linktools.errors import GitDivergedError
from linktools.git import GitRepository, GitSyncPolicy, get_git_unavailable_reason, is_git_available

from ..container import ContainerError

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


RepoGitResult = namedtuple("RepoGitResult", ["success", "revision", "dirty", "error"])


class RepoGit(object):
    """Dulwich-backed git operations, gated on runtime availability.

    One instance is shared for a Manager's whole lifetime (``RepoService.git``
    off the ``manager.repos`` cached property), so an unavailable-git warning
    is only ever emitted once per command run, however many repositories it
    touches.
    """

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager
        self._warned = False

    @property
    def logger(self):
        return self.manager.logger

    @property
    def available(self) -> bool:
        return is_git_available()

    def _warning_message(self, action: str) -> str:
        reason = get_git_unavailable_reason() or "Git support is unavailable."
        return (
            "%s cannot be completed: %s "
            "Local repository features remain available." % (action, reason)
        )

    def warn_unavailable(self, action: str) -> None:
        if self._warned:
            return
        self._warned = True
        self.logger.warning(self._warning_message(action))

    def clone(self, url: str, repo_path: str, branch: str = None) -> None:
        if not self.available:
            self.warn_unavailable("Cloning a Git repository")
            raise ContainerError(self._warning_message("Cloning a Git repository"))

        GitRepository.clone(self.manager.environ, url, repo_path, branch)

    def update(self, url: str, repo_path: str, branch: str = None, reset: bool = False) -> "RepoGitResult":
        if not self.available:
            self.warn_unavailable("Updating Git repositories")
            return RepoGitResult(
                success=False, revision=None, dirty=None,
                error=self._warning_message("Updating Git repositories"),
            )

        if not os.path.exists(repo_path):
            self.logger.info("Update git repository: %s" % url)
            GitRepository.clone(self.manager.environ, url, repo_path, branch)
            with GitRepository(self.manager.environ, repo_path) as repo:
                return RepoGitResult(success=True, revision=repo.head_sha(),
                                      dirty=repo.is_dirty(), error=None)

        repo = GitRepository.open_if_valid(self.manager.environ, repo_path)
        if repo is None:
            # Not a git repository (e.g. a local symlinked repo) -- nothing to sync.
            return RepoGitResult(success=True, revision=None, dirty=None, error=None)

        with repo:
            self.logger.info("Update git repository: %s" % url)
            if branch:
                repo.checkout_or_create(branch)
            try:
                repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE if reset
                          else GitSyncPolicy.STASH_AND_RESTORE)
            except GitDivergedError:
                # Without --force, a diverged branch must fail loudly and
                # leave local commits alone -- silently retrying with
                # RESET_TO_REMOTE here would discard unpushed work the user
                # never agreed to lose.
                raise GitDivergedError(
                    "Repository `%s` has diverged from the remote. "
                    "Re-run with --force to reset to remote." % url
                ) from None
            return RepoGitResult(success=True, revision=repo.head_sha(),
                                  dirty=repo.is_dirty(), error=None)

    def inspect(self, repo_path: "Any") -> "dict":
        if not self.available:
            self.warn_unavailable("Reading Git repository metadata")
            return {
                "applicable": True,
                "supported": False,
                "revision": None,
                "dirty": None,
                "reason": self._warning_message("Reading Git repository metadata"),
            }

        if not repo_path or not os.path.exists(repo_path):
            return {"applicable": True, "supported": True, "revision": None, "dirty": None, "reason": None}

        repo = GitRepository.open_if_valid(self.manager.environ, repo_path)
        if repo is None:
            return {"applicable": True, "supported": True, "revision": None, "dirty": None,
                     "reason": "Not a git repository."}

        with repo:
            # A corrupted object store/index, or a permission error, must
            # never propagate out of a read-only inspection -- describe()/
            # validate() must be able to report every OTHER repository even
            # when this one's Git state is broken, matching the contract
            # every other structured-error path in this module already
            # honors (RepoService.describe() never raises for one bad repo).
            try:
                revision = repo.head_sha()
                dirty = repo.is_dirty()
            except Exception as exc:
                return {"applicable": True, "supported": False, "revision": None, "dirty": None,
                         "reason": "Failed to read Git repository metadata: %s" % exc}
            return {"applicable": True, "supported": True, "revision": revision,
                     "dirty": dirty, "reason": None}
