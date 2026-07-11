#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repo configuration store.

Read/add/update/remove the INSTALLED_REPOS store and manage the on-disk repo
clone/symlink layout. Git sync is delegated to RepoSync.
"""
import os
import shutil
from typing import TYPE_CHECKING

from linktools import utils
from linktools.decorator import cached_property

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

    @cached_property
    def _repo_path(self):
        path = self.manager.data_path.joinpath("repo")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self) -> "dict[str, dict[str, str]]":
        # A failed migration is not cached, so retry it on every access.
        self.manager._migrated
        return self.manager._persistent_store.get(_REPO_KEY, {})

    def _dump(self, repos: "dict[str, dict[str, str]]") -> None:
        self.manager._migrated
        self.manager._persistent_store.set(_REPO_KEY, repos)

    def get_all(self) -> "dict[str, dict[str, str]]":
        return self._load()

    def add(self, url: str, branch: str = None, force: bool = False) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            # See InstalledStateStore.add for why this reload is necessary:
            # the lock alone doesn't stop this read-modify-write from
            # clobbering a concurrent writer's change with stale data.
            self.manager._persistent_store.reload()
            repos = self._load()

            def ensure_repo_not_exist(key):
                if key not in repos:
                    return
                if not force:
                    raise ContainerError(f"Repository `{key}` already exists.")
                self._remove_repo_file(repos.pop(key))
                self._dump(repos)

            if url.startswith(_GIT_PREFIXES):
                ensure_repo_not_exist(url)
                self.logger.info(f"Add git repository: {url}")
                repo_name = utils.guess_file_name(url)
                repo_path = self._choose_repo_path(repo_name)
                self.sync.clone_git(url, repo_path, branch)
                self._validate_new_repo_manifest(repo_path)
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
                self._validate_new_repo_manifest(repo_path)
                repos[path] = dict(type="local", repo_path=repo_path, repo_name=repo_name)

            self._dump(repos)

    def update(self, branch: str = None, reset: bool = False) -> None:
        for url, meta in self.get_all().items():
            self.sync.sync(url, meta, branch=branch, reset=reset)
            self._warn_if_manifest_incompatible_after_update(url, meta)

    def _warn_if_manifest_incompatible_after_update(self, url: str, meta: "dict[str, str]") -> None:
        # Spec section 26: re-read and validate the manifest after update
        # completes. No automatic Git rollback / transactional replace is
        # implemented here (explicitly deferred) -- this only informs.
        from .manifest import RepositoryManifestError
        repo_path = meta.get("repo_path")
        if not repo_path or not os.path.exists(repo_path):
            return
        try:
            manifest = self.manager.repo_manifest.load(repo_path)
        except RepositoryManifestError as exc:
            self.logger.warning(f"Repository `{url}` manifest is invalid after update: {exc}")
            return
        if manifest is None:
            return
        issues = self.manager.repo_manifest.check_host_requirements(manifest)
        if issues:
            details = "; ".join(issue.message for issue in issues)
            self.logger.warning(f"Repository `{url}` is incompatible with this host after update: {details}")

    def remove(self, url: str) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            self.manager._persistent_store.reload()
            repos = self._load()
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            self._remove_repo_file(repos.pop(url))
            self._dump(repos)

    def _validate_new_repo_manifest(self, repo_path: str) -> None:
        # Read .linktools.json (if any) and check host requirements before
        # this repo is ever written to INSTALLED_REPOS; on failure, clean up
        # the just-cloned/linked path rather than leaving a half-added repo
        # (Spec section 26). The full manifest is intentionally not persisted
        # into INSTALLED_REPOS itself, to avoid stale metadata drifting from
        # the on-disk .linktools.json.
        try:
            manifest = self.manager.repo_manifest.load(repo_path)
            self.manager.repo_manifest.ensure_loadable(manifest)
        except Exception:
            self._remove_repo_file(dict(repo_path=repo_path))
            raise

    def _choose_repo_path(self, name: str) -> str:
        index = 0
        path = os.path.join(self._repo_path, name)
        while os.path.lexists(path):
            path = os.path.join(self._repo_path, f"{name}_{index}")
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
