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
from urllib.parse import urlsplit

from linktools import utils
from linktools.core import ProjectProfile
from linktools.decorator import cached_property
from linktools.errors import ConfigError, ConfigValidationError

from ..container import ContainerError
from ...capabilities.cntr import __cap_cntr__
from .git import RepoGit
from .requirements import ensure_requirement

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


_REPO_KEY = "INSTALLED_REPOS"

# Recognized schemes for a remote Git URL. SCP-like addresses
# (`git@host:repo.git`, `user@host:path`) are deliberately not recognized --
# urlsplit has no scheme for them (the `@` isn't a valid scheme character),
# so they are left as local paths rather than risk misrouting one into Git
# cloning.
_GIT_SCHEMES = frozenset(("http", "https", "ssh", "git", "file"))

_REPO_TYPE_GIT = "git"
_REPO_TYPE_LOCAL = "local"
_SUPPORTED_REPO_TYPES = frozenset((_REPO_TYPE_GIT, _REPO_TYPE_LOCAL))


def _is_remote_git_url(value: "Any") -> bool:
    """Whether ``value`` is a Git URL this module clones/updates via
    RepoGit, as opposed to a local filesystem path it symlinks in as-is.

    Only an explicit recognized scheme counts -- a Windows drive path
    (``C:\\repo``, ``C:/repo``, parsed by ``urlsplit`` as scheme ``"c"``), a
    UNC path (``\\\\server\\share``), or a relative/absolute local path never
    matches one, so they always fall through to the local-repository branch
    below.
    """
    return isinstance(value, str) and urlsplit(value).scheme.lower() in _GIT_SCHEMES


def _reject_credential_url(url: str) -> None:
    """Reject an HTTP/HTTPS repository URL carrying embedded credentials
    (``https://user:token@host/repo``, ``https://token@host/repo``) -- the
    full URL (with credentials) is what gets persisted to settings.json,
    shown in ``repo list``/``status``/``validate`` output and error
    messages, and (until this fix) recorded into the Artifact Index. None
    of those are secret-aware, so a credential embedded in the URL itself
    would otherwise leak through all of them.

    ``ssh://git@host/repo.git`` is explicitly allowed: an SSH username is
    routing information, not a secret, and cloning over SSH already
    depends on key-based auth outside the URL entirely.
    """
    parsed = urlsplit(url)
    if parsed.scheme in ("http", "https") and (
            parsed.username is not None or parsed.password is not None):
        raise ContainerError(
            "Repository URL must not contain credentials. "
            "Use a credential helper or a credential-free URL."
        )


def safe_display_url(url: "Any") -> "Any":
    """``url`` with any embedded HTTP/HTTPS userinfo (``user:pass@``/
    ``token@``) stripped -- a no-op for a normal URL/local path, and for an
    SSH URL's username (``ssh://git@host/repo.git``), which is routing
    information rather than a secret -- see ``_reject_credential_url``.

    ``add()`` rejects a new credential-bearing HTTP/HTTPS URL outright, but
    an entry added before that check existed may still have one persisted
    -- every place a URL is shown to the user (``repo list``/``status``/
    ``validate``/``update`` output, error messages) must route through this
    instead of the raw value, so a legacy credential is never echoed back.
    """
    if not isinstance(url, str):
        return url
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return url
    if parsed.username is None and parsed.password is None:
        return url
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


def _compute_manager_config_keys(env_config: "Any") -> "frozenset[str]":
    keys = set()
    for field in env_config.schema.fields():
        keys.add(field.name)
        keys.update(field.aliases)
    return frozenset(keys)


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
        return self.manager.settings.get(_REPO_KEY, {})

    def _dump(self, repos: "dict[str, dict[str, str]]") -> None:
        self.manager.settings.set(_REPO_KEY, repos)

    def get_all(self) -> "dict[str, dict[str, str]]":
        """Every configured repository, as an independent copy -- `_load()`
        returns the persistent store's own live in-memory dict by
        reference, so a caller mutating the result (or one of its per-repo
        metadata dicts) directly must never be able to corrupt the store's
        cache out from under a future ``add()``/``remove()``."""
        return {url: dict(meta) for url, meta in self._load().items()}

    def add(self, url: str, branch: str = None, replace: bool = False) -> None:
        """Add a repository. ``replace=True`` allows replacing an already-
        added repository at the same key (URL or local path) -- otherwise
        an existing entry is a hard error.

        Transactional: the OLD repository's directory and Store entry are
        left completely untouched until the NEW repository has cloned (or
        symlinked) successfully, passed its requirement check, AND the
        Store has been durably updated to point at it. Only after all of
        that succeeds is the old directory removed -- and a failure to
        remove it is a warning, not a rollback of the now-successful
        replace. Any failure before that point (clone, requirement check,
        or the Store write itself) leaves the old repository exactly as it
        was and cleans up only the new (never-finished) directory.
        """
        _reject_credential_url(url)
        with self.manager.environ.locks.process_lock("cntr:repo"):
            # See InstalledStateStore.add for why this reload is necessary:
            # the lock alone doesn't stop this read-modify-write from
            # clobbering a concurrent writer's change with stale data.
            self.manager.settings.reload()
            repos = self._load()

            if _is_remote_git_url(url):
                key = url
            else:
                key = os.path.abspath(os.path.expanduser(url))

            existing = repos.get(key)
            if existing is not None and not replace:
                raise ContainerError(f"Repository `{key}` already exists.")

            if _is_remote_git_url(url):
                self.logger.info(f"Add git repository: {url}")
                repo_name = utils.guess_file_name(url)
                # _choose_repo_path always returns a path distinct from any
                # existing directory (including the one being replaced,
                # which still exists on disk at this point) -- the new
                # repository never reuses the old one's staging area.
                repo_path = self._choose_repo_path(repo_name)
                try:
                    self.git.clone(url, repo_path, branch)
                    self._validate_new_repo_requirement(repo_path)
                except Exception:
                    self._cleanup_failed_add(repo_path)
                    raise
                new_entry = dict(type=_REPO_TYPE_GIT, repo_path=repo_path, repo_name=repo_name)
            else:
                path = key
                if not os.path.exists(path) or not os.path.isdir(path):
                    raise ContainerError(f"Invalid local path: {url}")

                self.logger.info(f"Add local repository: {path}")
                repo_name = utils.guess_file_name(path)
                repo_path = self._choose_repo_path(repo_name)
                os.symlink(path, repo_path, target_is_directory=True)
                try:
                    self._validate_new_repo_requirement(repo_path)
                except Exception:
                    self._cleanup_failed_add(repo_path)
                    raise
                new_entry = dict(type=_REPO_TYPE_LOCAL, repo_path=repo_path, repo_name=repo_name)

            # A fresh dict, never a mutation of the one `_load()` returned --
            # that dict is the store's own live in-memory cache (ConfigStore
            # .get() returns it by reference, not a copy), so mutating it in
            # place would make the new entry "visible" via get_all() even if
            # the _dump() below never actually persists it.
            new_repos = dict(repos)
            new_repos[key] = new_entry
            try:
                self._dump(new_repos)
            except Exception:
                self._cleanup_failed_add(new_entry["repo_path"])
                raise

            if existing is not None:
                try:
                    self._remove_repo_file(existing)
                except Exception as exc:  # noqa: BLE001 - never undo a successful replace over cleanup
                    self.logger.warning(
                        f"Repository `{key}` replaced, but failed to remove its old "
                        f"directory `{existing.get('repo_path')}`: {exc}")

    def _cleanup_failed_add(self, repo_path: str) -> None:
        """The single cleanup path for a repository ``add()`` never
        finished (clone succeeded but requirement validation failed, clone
        itself failed after creating a partial directory, ...) -- never
        leaves a half-added directory behind, and never lets a cleanup
        failure hide the original error that triggered it."""
        if not os.path.lexists(repo_path):
            return
        try:
            self._remove_repo_file(dict(repo_path=repo_path))
        except Exception as cleanup_exc:  # noqa: BLE001 - never mask the original error
            self.logger.warning(f"Failed to clean repository directory {repo_path}: {cleanup_exc}")

    def _validate_repo_root(self, repo_path: "str | None") -> str:
        """Fail closed on any repository root that isn't a genuinely usable
        directory: missing, nonexistent, a dangling symlink, or a non-
        directory. The single choke point every caller that's about to read
        from or Git-sync a repository root routes through -- so "the root
        is unusable" is always an explicit, typed failure (never silently
        treated as compatible/updated, never a fallback to
        ``ProjectProfile(os.path.join(os.getcwd(), '.linktools.json'))`` --
        which would read the process's CWD instead of this repository).

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
            display_url = safe_display_url(url)

            # A repository persisted before credential URLs were rejected at
            # add() time -- fail closed instead of syncing it (which would
            # use the embedded credential), and never echo it back.
            if _is_remote_git_url(url):
                try:
                    _reject_credential_url(url)
                except ContainerError:
                    results.append(RepoUpdateResult(
                        url=display_url, updated=False, revision=None, compatible=False,
                        error="Repository URL contains embedded credentials from a "
                              "previous version. Remove and re-add it with a "
                              "credential-free URL.",
                    ))
                    continue

            try:
                repo_path = self._validate_repo_root(meta.get("repo_path"))
            except ContainerError as exc:
                results.append(RepoUpdateResult(url=display_url, updated=False, revision=None,
                                                 compatible=False, error=str(exc)))
                continue

            try:
                repo_type = self._get_repo_type(url, meta)

                if repo_type == _REPO_TYPE_GIT:
                    git_result = self.git.update(url, repo_path, branch=branch, reset=reset)
                    if not git_result.success:
                        results.append(RepoUpdateResult(url=display_url, updated=False, revision=git_result.revision,
                                                         compatible=False, error=git_result.error))
                        continue
                    revision = git_result.revision
                else:
                    revision = None

                compatible, error = self._revalidate_after_update(repo_path)
                results.append(RepoUpdateResult(
                    url=display_url, updated=True, revision=revision, compatible=compatible, error=error,
                ))
            except Exception as exc:  # noqa: BLE001 - one repo's sync failure must not hide the rest
                results.append(RepoUpdateResult(url=display_url, updated=False, revision=None,
                                                 compatible=False, error=str(exc)))
        return results

    def _revalidate_after_update(self, repo_path: "str | None") -> "tuple[bool, str | None]":
        try:
            repo_path = self._validate_repo_root(repo_path)
            file_config = ProjectProfile.for_root(repo_path)
            ensure_requirement(file_config, "linktools-cntr", __cap_cntr__.version)
        except ConfigValidationError as exc:
            return False, f"incompatible with this host after update: {exc}"
        except ConfigError as exc:
            return False, f".linktools.json is invalid after update: {exc}"
        except ContainerError as exc:
            return False, str(exc)
        return True, None

    def remove(self, url: str) -> None:
        with self.manager.environ.locks.process_lock("cntr:repo"):
            self.manager.settings.reload()
            repos = self._load()
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            # Directory deletion must succeed BEFORE the Store entry is
            # dropped -- a failure here (permission error, an in-use file)
            # must propagate and leave the Store still pointing at whatever
            # is left on disk, not silently lose track of it.
            self._remove_repo_file(repos[url])
            # A fresh dict, not a mutation of `_load()`'s live reference --
            # see add()'s identical concern.
            new_repos = dict(repos)
            del new_repos[url]
            self._dump(new_repos)

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
            url=safe_display_url(url),
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

        # A repository persisted before credential URLs were rejected at
        # add() time -- fail closed instead of ever touching it (cloning,
        # reading its .linktools.json, inspecting Git), and never echo the
        # credential back; the user must remove and re-add it.
        if _is_remote_git_url(url):
            try:
                _reject_credential_url(url)
            except ContainerError:
                info["repository_error"] = (
                    "Repository URL contains embedded credentials from a previous "
                    "version. Remove and re-add it with a credential-free URL."
                )
                info["git"] = {
                    "applicable": True, "supported": False, "revision": None,
                    "dirty": None, "reason": info["repository_error"],
                }
                return info

        # The repository root itself is validated before anything ever
        # reads from it -- a missing/dangling/non-directory root must never
        # fall through to ``ProjectProfile(os.path.join(os.getcwd(), '.linktools.json'))``,
        # which would silently read the *process's* CWD instead of this
        # repository.
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
        info["local_config"] = "present" if os.path.lexists(ProjectProfile.local_path(repo_path)) else "absent"

        try:
            file_config = ProjectProfile.for_root(repo_path)
            info["requires"] = dict(file_config.get("requires", {}))
            info["ignored_environment_keys"] = self._find_reserved_environment_keys(file_config)
            ensure_requirement(file_config, "linktools-cntr", __cap_cntr__.version)
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
        """Manager-owned keys (``_compute_manager_config_keys``) this
        repository's own ``.linktools.json`` declares under ``environment``
        -- every container resolves fields through the manager's own shared
        ``env_config``, so a repository's local ``environment`` section is
        never actually consulted for these (or any other key);
        ``describe()``/``validate()`` surface them as a warning instead of
        leaving the mismatch invisible. Never rejects the repository over
        this -- an existing file with a reserved key must keep loading.
        """
        local_environment = file_config.get("environment", {})
        reserved = _compute_manager_config_keys(self.manager.env_config)
        return sorted(key for key in local_environment if key in reserved)

    def validate(self, url: str = None) -> "tuple[dict[str, Any], list[str]]":
        """Describe one or all repositories; also report which are incompatible.

        Keyed by the credential-free display URL (``describe()`` already
        redacts everything inside each value) -- never the raw persisted
        key, which may still carry a legacy embedded credential.
        """
        repos = self.get_all()
        if url is not None:
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            targets = {url: repos[url]}
        else:
            targets = repos

        results = {safe_display_url(u): self.describe(u, m) for u, m in targets.items()}
        incompatible = sorted(u for u, info in results.items() if info.get("compatible") is False)
        return results, incompatible

    def _validate_new_repo_requirement(self, repo_path: str) -> None:
        # Read .linktools.json (if any) and check requires.linktools-cntr
        # before this repo is ever written to INSTALLED_REPOS. Cleanup of a
        # failed add is centralized in add()/_cleanup_failed_add -- this
        # only raises, so the clone-failure and requirement-failure paths
        # share one cleanup responsibility instead of two. The resolved
        # file config is intentionally not persisted into INSTALLED_REPOS
        # itself, to avoid stale metadata drifting from the on-disk
        # .linktools.json.
        try:
            file_config = ProjectProfile.for_root(repo_path)
            ensure_requirement(file_config, "linktools-cntr", __cap_cntr__.version)
        except Exception as exc:
            raise ContainerError(f"Repository `{repo_path}` is not usable: {exc}") from exc

    def _choose_repo_path(self, name: str) -> str:
        index = 0
        path = os.path.join(self._repo_path, name)
        while os.path.lexists(path):
            path = os.path.join(self._repo_path, f"{name}_{index}")
            index += 1
        return path

    def _remove_repo_file(self, repo: "dict[str, str]") -> None:
        """Remove a repository's on-disk directory/symlink -- raises on
        failure instead of swallowing it (no ``ignore_errors=True``), so a
        caller that must not lose track of a repo it failed to delete (see
        ``remove()``) gets a real exception to propagate. A caller that
        intends a failure here to be non-fatal (see ``add()``'s cleanup of
        an old, already-replaced repository) wraps this call itself."""
        repo_path = repo.get("repo_path", None)
        if repo_path and os.path.lexists(repo_path):
            if os.path.islink(repo_path):
                self.logger.info(f"Remove link {repo_path}")
                os.unlink(repo_path)
            elif os.path.isdir(repo_path):
                self.logger.info(f"Remove directory {repo_path}")
                shutil.rmtree(repo_path)
            else:
                self.logger.info(f"Remove file {repo_path}")
                os.remove(repo_path)
