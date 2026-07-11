#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LockStore (Spec Part VI): builds, writes, and loads
``<data_path>/container.lock.json`` -- the local, not-committed-downstream
deployment lock. Fully opt-in (section 43): nothing here is called by
up/restart/down, and a missing lock file is never a warning.
"""
import hashlib
import os
from typing import TYPE_CHECKING

from ..artifacts.index import collect_candidates, sha256_of
from ..artifacts.writer import atomic_write_text_if_changed
from ..repo.manifest import RepositoryManifestError, RepositoryManifestService
from .model import DeploymentLock, LockedArtifact, LockedContainer, LockedRepository

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager

LOCK_SCHEMA_VERSION = 1
LOCK_FILE_NAME = "container.lock.json"

# Static inputs ContainerLoader actually reads per directory (Spec section 41);
# deliberately not "every file" -- large data/download/app dirs are never hashed.
_STATIC_INPUT_NAMES = frozenset({RepositoryManifestService.file_name, "container.py", "Dockerfile"})


def _local_repo_definitions_sha256(repo_path: str, docker_compose_names, max_level: int = 2) -> "str | None":
    """Hash the static inputs the loader reads across a local repo (root +
    up to max_level of subdirectories), in a stable (sorted, relative-path)
    order. Returns None if the repo can't be walked -- callers then mark
    the lock entry reproducible=False rather than faking a hash.

    Besides the fixed filenames, every file under a ``templates`` directory
    is included: containers render arbitrary config templates (nginx/authelia/
    lldap all ship one) via ``render_template(self.get_source_path("templates",
    ...))``, so those are "same-directory template files" per spec section 41
    even though their names aren't known in advance.
    """
    static_names = _STATIC_INPUT_NAMES | set(docker_compose_names)
    try:
        hasher = hashlib.sha256()
        for dir_path, level in _walk_levels(repo_path, max_level):
            in_templates_dir = os.path.basename(dir_path) == "templates"
            for name in sorted(os.listdir(dir_path)):
                if name not in static_names and not in_templates_dir:
                    continue
                full_path = os.path.join(dir_path, name)
                if not os.path.isfile(full_path):
                    continue
                rel_path = os.path.relpath(full_path, repo_path)
                with open(full_path, "rb") as f:
                    content = f.read()
                hasher.update(rel_path.replace(os.sep, "/").encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(content)
        return hasher.hexdigest()
    except OSError:
        return None


def _walk_levels(path: str, max_level: int):
    if not os.path.isdir(path):
        return
    yield path, 0
    if max_level <= 0:
        return
    for name in sorted(os.listdir(path)):
        child = os.path.join(path, name)
        if os.path.isdir(child):
            yield from _walk_levels(child, max_level - 1)


class LockStore:

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    @property
    def path(self) -> str:
        return os.path.join(str(self.manager.data_path), LOCK_FILE_NAME)

    def build(self) -> "DeploymentLock":
        lock, _candidates = self._build_with_candidates()
        return lock

    def build_and_preflight(self) -> "tuple[DeploymentLock, str]":
        """Build the lock and run Compose preflight against the exact same
        rendered candidates (Spec section 42), in one pass -- so `ct-cntr
        lock` never prepares containers or renders candidates twice."""
        lock, candidates = self._build_with_candidates()
        candidate_files = {dest: content for dest, (_kind, _name, content) in candidates.items()}
        preflight = self.manager.docker_inspector.preflight_candidates(candidate_files) if candidate_files else "skipped"
        return lock, preflight

    def _build_with_candidates(self) -> "tuple[DeploymentLock, dict]":
        from ...capabilities.cntr import __cap_cntr__
        manager = self.manager

        containers = manager.prepare_installed_containers()
        candidates = collect_candidates(manager, containers)
        artifacts = {
            os.path.relpath(dest, str(manager.data_path)).replace(os.sep, "/"):
                LockedArtifact(kind=kind, sha256=sha256_of(content))
            for dest, (kind, _container_name, content) in candidates.items()
        }

        lock = DeploymentLock(
            schema_version=LOCK_SCHEMA_VERSION,
            project=manager.project_name,
            linktools_cntr=__cap_cntr__.version,
            repositories=tuple(self._build_repository_locks()),
            containers=tuple(self._build_container_locks(containers)),
            artifacts=artifacts,
        )
        return lock, candidates

    def _build_repository_locks(self) -> "list[LockedRepository]":
        manager = self.manager
        locks = []
        for url, meta in manager.repo_store.get_all().items():
            repo_type = meta.get("type", "unknown")
            repo_path = meta.get("repo_path")

            manifest_sha256 = None
            manifest_version = None
            if repo_path and os.path.exists(repo_path):
                manifest_path = os.path.join(repo_path, RepositoryManifestService.file_name)
                if os.path.exists(manifest_path):
                    with open(manifest_path, "rb") as f:
                        manifest_sha256 = hashlib.sha256(f.read()).hexdigest()
                    try:
                        manifest = manager.repo_manifest.load(repo_path)
                    except RepositoryManifestError:
                        manifest = None
                    if manifest is not None:
                        manifest_version = manifest.version

            if repo_type == "git":
                branch = revision = None
                if repo_path and os.path.exists(repo_path):
                    try:
                        from linktools.git import GitRepository
                        repo = GitRepository(manager.environ, repo_path)
                        branch = repo.current_branch()
                        revision = repo.head_sha()
                    except Exception:  # noqa: BLE001 - lock build must never crash on one bad repo
                        pass
                locks.append(LockedRepository(
                    url=url, type="git", branch=branch, revision=revision,
                    manifest_sha256=manifest_sha256, manifest_version=manifest_version,
                    definitions_sha256=None, reproducible=revision is not None,
                ))
            else:
                definitions_sha256 = None
                if repo_path and os.path.exists(repo_path):
                    definitions_sha256 = _local_repo_definitions_sha256(
                        repo_path, manager.docker_compose_names)
                locks.append(LockedRepository(
                    url=url, type=repo_type, branch=None, revision=None,
                    manifest_sha256=manifest_sha256, manifest_version=manifest_version,
                    definitions_sha256=definitions_sha256, reproducible=definitions_sha256 is not None,
                ))
        return locks

    def _build_container_locks(self, containers) -> "list[LockedContainer]":
        locks = []
        for container in containers:
            repository = getattr(container, "_repository", None)
            repository_url = repository.url if repository is not None and not repository.builtin else None
            locks.append(LockedContainer(
                name=container.name,
                repository_url=repository_url,
                services=tuple(container.services.keys()),
            ))
        return locks

    # -- (de)serialization: canonical JSON, no secrets ------------------------

    def to_dict(self, lock: "DeploymentLock") -> "dict[str, Any]":
        return dict(
            schema_version=lock.schema_version,
            project=lock.project,
            linktools_cntr=lock.linktools_cntr,
            repositories=[
                dict(
                    url=r.url, type=r.type, branch=r.branch, revision=r.revision,
                    manifest_sha256=r.manifest_sha256, manifest_version=r.manifest_version,
                    definitions_sha256=r.definitions_sha256, reproducible=r.reproducible,
                )
                for r in lock.repositories
            ],
            containers=[
                dict(name=c.name, repository_url=c.repository_url, services=list(c.services))
                for c in lock.containers
            ],
            artifacts={
                path: dict(kind=a.kind, sha256=a.sha256)
                for path, a in lock.artifacts.items()
            },
        )

    def from_dict(self, data: "dict[str, Any]") -> "DeploymentLock":
        return DeploymentLock(
            schema_version=data.get("schema_version", LOCK_SCHEMA_VERSION),
            project=data.get("project", ""),
            linktools_cntr=data.get("linktools_cntr", ""),
            repositories=tuple(
                LockedRepository(
                    url=r.get("url"), type=r.get("type"), branch=r.get("branch"), revision=r.get("revision"),
                    manifest_sha256=r.get("manifest_sha256"), manifest_version=r.get("manifest_version"),
                    definitions_sha256=r.get("definitions_sha256"), reproducible=r.get("reproducible", True),
                )
                for r in data.get("repositories", [])
            ),
            containers=tuple(
                LockedContainer(name=c.get("name"), repository_url=c.get("repository_url"),
                                services=tuple(c.get("services", ())))
                for c in data.get("containers", [])
            ),
            artifacts={
                path: LockedArtifact(kind=entry.get("kind"), sha256=entry.get("sha256"))
                for path, entry in data.get("artifacts", {}).items()
            },
        )

    def write(self, lock: "DeploymentLock") -> bool:
        import json
        content = json.dumps(self.to_dict(lock), sort_keys=True, indent=2) + "\n"
        return atomic_write_text_if_changed(self.path, content)

    def load(self) -> "DeploymentLock | None":
        import json
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return self.from_dict(data)
