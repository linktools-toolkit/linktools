#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository business entry point: the configured repository set
(add/update/remove/list), local-path and Git-backed repositories,
``requires.linktools-cntr`` gating, and read-only describe/validate.

Dulwich-backed work is delegated entirely to
:class:`~linktools.cntr.repo.git.RepoGit`; this module never imports dulwich.
"""
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools import utils
from linktools.core import LinktoolsFileConfigLoader, ensure_requirement
from linktools.decorator import cached_property
from linktools.errors import ConfigError, ConfigValidationError

from ..container import ContainerError
from ...capabilities.cntr import __cap_cntr__
from .git import RepoGit

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


_REPO_KEY = "INSTALLED_REPOS"
_GIT_PREFIXES = ("http://", "https://", "ssh://", "git@")


@dataclass(frozen=True)
class RepoUpdateResult:
    url: str
    updated: bool
    revision: "str | None"
    compatible: bool
    error: "str | None"


class RepoService(object):
    """Owns the configured repository set behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager
        self.git = RepoGit(manager)

    @property
    def logger(self):
        return self.manager.logger

    @cached_property
    def _repo_path(self):
        path = self.manager.data_path.joinpath("repo")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load(self) -> "dict[str, dict[str, str]]":
        return self.manager._persistent_store.get(_REPO_KEY, {})

    def _dump(self, repos: "dict[str, dict[str, str]]") -> None:
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
                self.git.clone(url, repo_path, branch)
                self._validate_new_repo_requirement(repo_path)
                repos[url] = dict(type="git", repo_path=repo_path, repo_name=repo_name)
            else:
                path = os.path.abspath(os.path.expanduser(url))
                if not os.path.exists(path) or not os.path.isdir(path):
                    raise ContainerError(f"Invalid local path: {url}")

                ensure_repo_not_exist(path)
                self.logger.info(f"Add local repository: {path}")
                repo_name = utils.guess_file_name(path)
                repo_path = self._choose_repo_path(repo_name)
                os.symlink(path, repo_path, target_is_directory=True)
                self._validate_new_repo_requirement(repo_path)
                repos[path] = dict(type="local", repo_path=repo_path, repo_name=repo_name)

            self._dump(repos)

    def update(self, branch: str = None, reset: bool = False) -> "list[RepoUpdateResult]":
        """Sync every repository, then re-check each one's requires.linktools-cntr.

        Never stops at the first failure or incompatibility -- every
        repository is synced and reported, so one repo's problem can never
        hide the state of the rest. Does not attempt a Git rollback on
        incompatibility: the repo is left updated (and, per
        ``ContainerLoader``, simply not loaded again until it's compatible);
        the caller decides whether any result should make the command exit
        non-zero.
        """
        results = []
        for url, meta in self.get_all().items():
            repo_path = meta.get("repo_path")
            if not repo_path:
                continue

            try:
                git_result = self.git.update(url, repo_path, branch=branch, reset=reset)
            except Exception as exc:  # noqa: BLE001 - one repo's sync failure must not hide the rest
                results.append(RepoUpdateResult(url=url, updated=False, revision=None,
                                                 compatible=False, error=str(exc)))
                continue

            if not git_result.success:
                results.append(RepoUpdateResult(url=url, updated=False, revision=git_result.revision,
                                                 compatible=False, error=git_result.error))
                continue

            compatible, error = self._revalidate_after_update(repo_path)
            results.append(RepoUpdateResult(
                url=url, updated=True, revision=git_result.revision, compatible=compatible, error=error,
            ))
        return results

    def _revalidate_after_update(self, repo_path: str) -> "tuple[bool, str | None]":
        if not repo_path or not os.path.exists(repo_path):
            return True, None

        try:
            file_config = self.manager.environ.load_file_config(local_root=repo_path)
            ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
        except ConfigValidationError as exc:
            return False, f"incompatible with this host after update: {exc}"
        except ConfigError as exc:
            return False, f".linktools.json is invalid after update: {exc}"
        return True, None

    def remove(self, url: str) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            self.manager._persistent_store.reload()
            repos = self._load()
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            self._remove_repo_file(repos.pop(url))
            self._dump(repos)

    def describe(self, url: str, meta: "dict[str, Any]") -> "dict[str, Any]":
        """Read-only status: local file config presence, declared
        ``requires.linktools-cntr``, compatibility, and Git revision/dirty
        state. Never imports the repository's ``container.py``."""
        repo_path = meta.get("repo_path")
        info: "dict[str, Any]" = dict(url=url, type=meta.get("type", "unknown"), repo_path=repo_path)

        local_path = (os.path.join(repo_path, LinktoolsFileConfigLoader.local_file_name)
                      if repo_path else None)
        if local_path and os.path.exists(local_path):
            info["local_config"] = "present"
            try:
                file_config = self.manager.environ.load_file_config(local_root=repo_path)
                info["requires"] = dict(file_config.local_config.requires)
                ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
                info["compatible"] = True
            except ConfigValidationError as exc:
                info["compatible"] = False
                info["compatibility_issues"] = [str(exc)]
            except ConfigError as exc:
                info["local_config_error"] = str(exc)
                info["compatible"] = False
        else:
            info["local_config"] = "absent"
            info["compatible"] = True

        info["git"] = self.git.inspect(repo_path)
        return info

    def validate(self, url: str = None) -> "tuple[dict[str, Any], list[str]]":
        """Describe one or all repositories; also report which are incompatible."""
        repos = self.get_all()
        if url is not None:
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            targets = {url: repos[url]}
        else:
            targets = repos

        results = {u: self.describe(u, m) for u, m in targets.items()}
        incompatible = sorted(u for u, info in results.items() if info.get("compatible") is False)
        return results, incompatible

    def _validate_new_repo_requirement(self, repo_path: str) -> None:
        # Read .linktools.json (if any) and check requires.linktools-cntr
        # before this repo is ever written to INSTALLED_REPOS; on failure,
        # clean up the just-cloned/linked path rather than leaving a
        # half-added repo. The resolved file config is intentionally not
        # persisted into INSTALLED_REPOS itself, to avoid stale metadata
        # drifting from the on-disk .linktools.json.
        try:
            file_config = self.manager.environ.load_file_config(local_root=repo_path)
            ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
        except Exception as exc:
            self._remove_repo_file(dict(repo_path=repo_path))
            raise ContainerError(f"Repository `{repo_path}` is not usable: {exc}") from exc

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
