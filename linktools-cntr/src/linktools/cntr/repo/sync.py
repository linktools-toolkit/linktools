#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Git / local repo synchronization: clone-on-demand for missing git repos,
stash-or-reset on dirty working trees, branch checkout/create, fast-forward
sync with a diverged-fallback, and stashed-change restore.
"""
import os
from typing import TYPE_CHECKING

from dulwich.errors import NotGitRepository

from linktools.errors import GitDivergedError
from linktools.git import GitRepository, GitSyncPolicy

if TYPE_CHECKING:
    from ..manager import ContainerManager


class RepoSync:
    """Clone and sync individual repositories behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    @property
    def environ(self):
        return self.manager.environ

    @property
    def logger(self):
        return self.manager.logger

    def clone_git(self, url: str, repo_path: str, branch: str = None) -> None:
        GitRepository.clone(self.environ, url, repo_path, branch)

    def link_local(self, source: str, repo_path: str) -> None:
        os.symlink(source, repo_path, target_is_directory=True)

    def sync(self, url: str, meta: dict, branch: str = None, reset: bool = False) -> None:
        repo_type = meta.get("type", None)
        repo_path = meta.get("repo_path", None)
        if not repo_path:
            return

        if repo_type == "git" and not os.path.exists(repo_path):
            self.logger.info(f"Update git repository: {url}")
            GitRepository.clone(self.environ, url, repo_path, branch)
            return

        if not os.path.exists(repo_path):
            return

        try:
            repo = GitRepository(self.environ, repo_path)
        except NotGitRepository:
            self.logger.debug(f"Invalid git repository, skip: {url}")
            return

        if repo_type == "git":
            self.logger.info(f"Update git repository: {url}")

        is_stash = False
        try:
            if repo.is_dirty():
                if not reset:
                    self.logger.info(f"Repository `{repo_path}` is dirty, stash changes before pull")
                    is_stash = True
                    repo.git.stash()
                else:
                    self.logger.warning(f"Repository `{repo_path}` is dirty, reset to HEAD")
                    repo.git.reset(hard=True)

            if branch:
                if branch in repo.heads:
                    self.logger.info(f"Checkout branch `{branch}` in repository `{repo_path}`")
                    repo.git.checkout(branch)
                else:
                    self.logger.info(f"Branch `{branch}` not found in repository `{repo_path}`, create and checkout")
                    new_branch = repo.create_head(branch)
                    new_branch.checkout()

            try:
                repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE if reset
                          else GitSyncPolicy.FAST_FORWARD_ONLY)
            except GitDivergedError:
                if reset:
                    raise
                self.logger.warning(
                    f"Repository `{url}` has diverged from the remote, force resetting ..."
                )
                repo.sync(policy=GitSyncPolicy.RESET_TO_REMOTE)

        finally:
            if is_stash:
                self.logger.info(f"Repository `{repo_path}` is updated, pop stashed changes")
                repo.git.stash("pop")
