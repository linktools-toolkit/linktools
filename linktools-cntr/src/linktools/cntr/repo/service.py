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

# Recognized protocol prefixes for a remote Git URL. SCP-like addresses
# (`git@host:repo.git`, `user@host:path`) are deliberately not recognized --
# they are indistinguishable from a bare local path without also parsing
# Windows drive letters/UNC prefixes, so they are left as local paths rather
# than risk misrouting one into Git cloning.
_GIT_PREFIXES = ("http://", "https://", "ssh://", "git://", "file://")

_REPO_TYPE_GIT = "git"
_REPO_TYPE_LOCAL = "local"
_SUPPORTED_REPO_TYPES = frozenset((_REPO_TYPE_GIT, _REPO_TYPE_LOCAL))


def _is_remote_git_url(value: "Any") -> bool:
    """Whether ``value`` is a Git URL this module clones/updates via
    RepoGit, as opposed to a local filesystem path it symlinks in as-is.

    Only explicit protocol prefixes count -- a Windows drive path
    (``C:\\repo``, ``C:/repo``), a UNC path (``\\\\server\\share``), or a
    relative/absolute local path never matches one, so they always fall
    through to the local-repository branch below.
    """
    return isinstance(value, str) and value.startswith(_GIT_PREFIXES)


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

            if _is_remote_git_url(url):
                ensure_repo_not_exist(url)
                self.logger.info(f"Add git repository: {url}")
                repo_name = utils.guess_file_name(url)
                repo_path = self._choose_repo_path(repo_name)
                self.git.clone(url, repo_path, branch)
                self._validate_new_repo_requirement(repo_path)
                repos[url] = dict(type=_REPO_TYPE_GIT, repo_path=repo_path, repo_name=repo_name)
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
                repos[path] = dict(type=_REPO_TYPE_LOCAL, repo_path=repo_path, repo_name=repo_name)

            self._dump(repos)

    def _validate_repo_root(self, repo_path: "str | None") -> str:
        """Fail closed on any repository root that isn't a genuinely usable
        directory: missing, nonexistent, a dangling symlink, or a non-
        directory. The single choke point every caller that's about to read
        from or Git-sync a repository root routes through -- so "the root
        is unusable" is always an explicit, typed failure (never silently
        treated as compatible/updated, never a fallback to
        ``load_file_config(local_root=None)`` -- which would read the
        process's CWD instead of this repository).

        Check order matters: lexists before the dangling-symlink check
        before isdir, so a dangling symlink is reported as exactly that
        rather than the less specific "not a directory".
        """
        if not repo_path:
            raise ContainerError("Repository path is missing.")

        path = os.path.abspath(os.path.expanduser(str(repo_path)))

        if not os.path.lexists(path):
            raise ContainerError(f"Repository path does not exist: {path}")

        if os.path.islink(path) and not os.path.exists(path):
            raise ContainerError(f"Repository link is dangling: {path}")

        if not os.path.isdir(path):
            raise ContainerError(f"Repository path is not a directory: {path}")

        return path

    def _get_repo_type(self, url: str, meta: "dict[str, Any]") -> str:
        """The repository's business type, as recorded in ``meta['type']``.

        The single place every update/describe/validate call routes a
        repository's type through, so a corrupt/unrecognized persisted
        ``type`` fails loudly here instead of each caller silently guessing
        "local" or "git" on its own.
        """
        repo_type = meta.get("type")
        if repo_type not in _SUPPORTED_REPO_TYPES:
            raise ContainerError(f"Repository `{url}` has an unsupported type: {repo_type!r}")
        return repo_type

    def update(self, branch: str = None, reset: bool = False) -> "list[RepoUpdateResult]":
        """Sync every repository, then re-check each one's requires.linktools-cntr.

        Never stops at the first failure or incompatibility -- every
        repository is synced and reported, so one repo's problem can never
        hide the state of the rest. Does not attempt a Git rollback on
        incompatibility: the repo is left updated (and, per
        ``ContainerLoader``, simply not loaded again until it's compatible);
        the caller decides whether any result should make the command exit
        non-zero.

        A local (non-git) repository is already linked straight to its
        source directory -- there is no Git sync step, so it never reaches
        ``RepoGit`` (and therefore never triggers a Git-unavailable warning).
        ``updated=True`` for a local repo means "this pass re-validated it
        successfully", not "its Git content changed".

        The repository root is validated *before* dispatching to Git, for
        every type -- ``RepoGit.update()`` itself would clone into a missing
        directory on demand, but ``update`` and ``add`` stay separate
        responsibilities here: a missing/dangling checkout root is always a
        hard failure the user re-``add``s, never an implicit self-heal.
        """
        results = []
        for url, meta in self.get_all().items():
            try:
                repo_path = self._validate_repo_root(meta.get("repo_path"))
            except ContainerError as exc:
                results.append(RepoUpdateResult(url=url, updated=False, revision=None,
                                                 compatible=False, error=str(exc)))
                continue

            try:
                repo_type = self._get_repo_type(url, meta)

                if repo_type == _REPO_TYPE_GIT:
                    git_result = self.git.update(url, repo_path, branch=branch, reset=reset)
                    if not git_result.success:
                        results.append(RepoUpdateResult(url=url, updated=False, revision=git_result.revision,
                                                         compatible=False, error=git_result.error))
                        continue
                    revision = git_result.revision
                else:
                    revision = None

                compatible, error = self._revalidate_after_update(repo_path)
                results.append(RepoUpdateResult(
                    url=url, updated=True, revision=revision, compatible=compatible, error=error,
                ))
            except Exception as exc:  # noqa: BLE001 - one repo's sync failure must not hide the rest
                results.append(RepoUpdateResult(url=url, updated=False, revision=None,
                                                 compatible=False, error=str(exc)))
        return results

    def _revalidate_after_update(self, repo_path: "str | None") -> "tuple[bool, str | None]":
        try:
            repo_path = self._validate_repo_root(repo_path)
            file_config = self.manager.environ.load_file_config(local_root=repo_path)
            ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
        except ConfigValidationError as exc:
            return False, f"incompatible with this host after update: {exc}"
        except ConfigError as exc:
            return False, f".linktools.json is invalid after update: {exc}"
        except ContainerError as exc:
            return False, str(exc)
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
        state. Never imports the repository's ``container.py``.

        Never raises for a single bad repository (an unsupported/corrupt
        persisted ``type``, a missing/dangling repository root, a dangling
        ``.linktools.json``, ...) -- every problem is reported inline in the
        returned dict, so one repo's problem can never hide the rest of a
        multi-repo ``status``/``validate`` call.
        """
        info: "dict[str, Any]" = dict(
            url=url,
            type=meta.get("type", "unknown"),
            repo_type=meta.get("type"),
            repo_name=meta.get("repo_name"),
            repo_path=meta.get("repo_path"),
            available=False,
            compatible=False,
            local_config="unknown",
            requires={},
            compatibility_issues=[],
        )

        # The repository root itself is validated before anything ever
        # reads from it -- a missing/dangling/non-directory root must never
        # fall through to ``load_file_config(local_root=None)``, which would
        # silently read the *process's* CWD instead of this repository.
        try:
            repo_path = self._validate_repo_root(meta.get("repo_path"))
        except ContainerError as exc:
            info["repository_error"] = str(exc)
            info["git"] = {
                "applicable": meta.get("type") == _REPO_TYPE_GIT,
                "supported": False,
                "revision": None,
                "dirty": None,
                "reason": str(exc),
            }
            return info

        try:
            repo_type = self._get_repo_type(url, meta)
        except ContainerError as exc:
            info["available"] = True
            info["local_config_error"] = str(exc)
            info["git"] = {"applicable": False, "supported": False, "revision": None,
                            "dirty": None, "reason": str(exc)}
            return info

        info["available"] = True

        # ``.linktools.json`` presence is reported as a raw filesystem fact
        # (lexists -- a dangling symlink counts as "present", not "absent")
        # separately from whether it actually loads: the real Loader (never
        # re-implemented here) is the only thing that decides load success,
        # so a present-but-broken file is reported as present + incompatible
        # rather than silently downgraded to absent.
        local_path = os.path.join(repo_path, LinktoolsFileConfigLoader.local_file_name)
        info["local_config"] = "present" if os.path.lexists(local_path) else "absent"

        try:
            file_config = self.manager.environ.load_file_config(local_root=repo_path)
            info["requires"] = dict(file_config.local_config.requires)
            info["ignored_environment_keys"] = self._find_reserved_environment_keys(file_config)
            ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
            info["compatible"] = True
        except ConfigValidationError as exc:
            info["compatible"] = False
            info["compatibility_issues"] = [str(exc)]
        except ConfigError as exc:
            info["local_config_error"] = str(exc)
            info["compatible"] = False

        if repo_type == _REPO_TYPE_GIT:
            info["git"] = self.git.inspect(repo_path)
        else:
            info["git"] = {
                "applicable": False,
                "supported": True,
                "revision": None,
                "dirty": None,
                "reason": "Local repository.",
            }
        return info

    def _find_reserved_environment_keys(self, file_config: "Any") -> "list[str]":
        """Manager-owned keys (``ContainerManager._get_manager_config_keys``)
        this repository's own ``.linktools.json`` declares under
        ``environment`` -- these are silently ignored by
        ``ManagerConfigSource`` (the manager's value always wins), so
        ``describe()``/``validate()`` surface them as a warning instead of
        leaving the mismatch invisible. Never rejects the repository over
        this -- an existing file with a reserved key must keep loading.
        """
        local_environment = file_config.local_config.environment
        reserved = self.manager._get_manager_config_keys()
        return sorted(key for key in local_environment if key in reserved)

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
