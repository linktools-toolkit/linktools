#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo configuration store (refactor spec Phase 4).

Extracted from ContainerManager: read/add/update/remove the INSTALLED_REPOS store
and manage the on-disk repo clone/symlink layout. Git sync is delegated to
RepoSync. Behavior is unchanged.
"""
import os
import shutil
from typing import TYPE_CHECKING

from linktools import utils

from ..container import ContainerError
from .sync import RepoSync

if TYPE_CHECKING:
    from ..manager import ContainerManager


_REPO_KEY = "INSTALLED_REPOS"
_GIT_PREFIXES = ("http://", "https://", "ssh://", "git@")


class RepoStore:
    """Owns the configured repository set behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager
        self.sync = RepoSync(manager)

    @property
    def logger(self):
        return self.manager.logger

    def get_all(self) -> "dict[str, dict[str, str]]":
        return self.manager._load_setting(_REPO_KEY, default={})

    def add(self, url: str, branch: str = None, force: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            repos = self.manager._load_setting(_REPO_KEY, reload=True, default={})

            def ensure_repo_not_exist(key):
                if key not in repos:
                    return
                if not force:
                    raise ContainerError(f"Repository `{key}` already exists.")
                self._remove_repo_file(repos.pop(key))
                self.manager._dump_setting(_REPO_KEY, repos)

            if url.startswith(_GIT_PREFIXES):
                ensure_repo_not_exist(url)
                self.logger.info(f"Add git repository: {url}")
                repo_name = utils.guess_file_name(url)
                repo_path = self._choose_repo_path(repo_name)
                self.sync.clone_git(url, repo_path, branch)
                repos[url] = dict(type="git", repo_path=repo_path, repo_name=repo_name)
            else:
                path = os.path.abspath(os.path.expanduser(url))
                if not os.path.exists(path) or not os.path.isdir(path):
                    raise ContainerError(f"Invalid local path: {url}")

                ensure_repo_not_exist(path)
                self.logger.info(f"Add local repository: {path}")
                repo_name = utils.guess_file_name(path)
                repo_path = self._choose_repo_path(repo_name)
                self.sync.link_local(path, repo_path)
                repos[path] = dict(type="local", repo_path=repo_path, repo_name=repo_name)

            self.manager._dump_setting(_REPO_KEY, repos)

    def update(self, branch: str = None, reset: bool = False) -> None:
        for url, meta in self.get_all().items():
            self.sync.sync(url, meta, branch=branch, reset=reset)

    def remove(self, url: str) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            repos = self.manager._load_setting(_REPO_KEY, reload=True, default={})
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            self._remove_repo_file(repos.pop(url))
            self.manager._dump_setting(_REPO_KEY, repos)

    def _choose_repo_path(self, name: str) -> str:
        index = 0
        path = os.path.join(self.manager._repo_path, name)
        while os.path.lexists(path):
            path = os.path.join(self.manager._repo_path, f"{name}_{index}")
            index += 1
        return path

    def _remove_repo_file(self, repo: "dict[str, str]") -> None:
        repo_path = repo.get("repo_path", None)
        if repo_path and os.path.lexists(repo_path):
            if os.path.islink(repo_path):
                self.logger.info(f"Remove link {repo_path}")
                os.unlink(repo_path)
            elif os.path.isdir(repo_path):
                self.logger.info(f"Remove directory {repo_path}")
                shutil.rmtree(repo_path, ignore_errors=True)
