#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment Lock diff: repo revision/manifest drift,
installed-container-set drift, and generated-artifact added/changed/removed
-- used by both ``lock --check`` (pass/fail) and ``ct-cntr diff`` (detail).
Never includes secrets or full config; only hashes/identifiers.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import DeploymentLock


@dataclass(frozen=True)
class RepositoryDrift:
    url: str
    field: str
    old: "str | None"
    new: "str | None"


@dataclass(frozen=True)
class ArtifactDrift:
    path: str
    change: str  # "added" | "changed" | "removed"
    old_sha256: "str | None"
    new_sha256: "str | None"


@dataclass(frozen=True)
class LockDiff:
    old_cntr_version: "str | None"
    new_cntr_version: "str | None"
    repository_drifts: "tuple[RepositoryDrift, ...]" = field(default_factory=tuple)
    repositories_added: "tuple[str, ...]" = field(default_factory=tuple)
    repositories_removed: "tuple[str, ...]" = field(default_factory=tuple)
    containers_added: "tuple[str, ...]" = field(default_factory=tuple)
    containers_removed: "tuple[str, ...]" = field(default_factory=tuple)
    artifact_drifts: "tuple[ArtifactDrift, ...]" = field(default_factory=tuple)

    @property
    def cntr_version_changed(self) -> bool:
        return self.old_cntr_version != self.new_cntr_version

    @property
    def is_empty(self) -> bool:
        return not (
            self.cntr_version_changed
            or self.repository_drifts
            or self.repositories_added
            or self.repositories_removed
            or self.containers_added
            or self.containers_removed
            or self.artifact_drifts
        )


_REPO_FIELDS = ("branch", "revision", "manifest_sha256", "manifest_version", "definitions_sha256")


def compute_diff(old: "DeploymentLock | None", new: "DeploymentLock") -> "LockDiff":
    if old is None:
        return LockDiff(
            old_cntr_version=None,
            new_cntr_version=new.linktools_cntr,
            repositories_added=tuple(sorted(r.url for r in new.repositories)),
            containers_added=tuple(sorted(c.name for c in new.containers)),
            artifact_drifts=tuple(
                ArtifactDrift(path=path, change="added", old_sha256=None, new_sha256=artifact.sha256)
                for path, artifact in sorted(new.artifacts.items())
            ),
        )

    old_repos = {r.url: r for r in old.repositories}
    new_repos = {r.url: r for r in new.repositories}

    repository_drifts = []
    for url in sorted(set(old_repos) & set(new_repos)):
        old_repo, new_repo = old_repos[url], new_repos[url]
        for field_name in _REPO_FIELDS:
            old_value, new_value = getattr(old_repo, field_name), getattr(new_repo, field_name)
            if old_value != new_value:
                repository_drifts.append(RepositoryDrift(url=url, field=field_name, old=old_value, new=new_value))

    old_containers = {c.name for c in old.containers}
    new_containers = {c.name for c in new.containers}

    artifact_drifts = []
    old_artifacts, new_artifacts = old.artifacts, new.artifacts
    for path in sorted(set(old_artifacts) | set(new_artifacts)):
        old_artifact, new_artifact = old_artifacts.get(path), new_artifacts.get(path)
        if old_artifact is None:
            artifact_drifts.append(ArtifactDrift(
                path=path, change="added", old_sha256=None, new_sha256=new_artifact.sha256))
        elif new_artifact is None:
            artifact_drifts.append(ArtifactDrift(
                path=path, change="removed", old_sha256=old_artifact.sha256, new_sha256=None))
        elif old_artifact.sha256 != new_artifact.sha256:
            artifact_drifts.append(ArtifactDrift(
                path=path, change="changed", old_sha256=old_artifact.sha256, new_sha256=new_artifact.sha256))

    return LockDiff(
        old_cntr_version=old.linktools_cntr,
        new_cntr_version=new.linktools_cntr,
        repository_drifts=tuple(repository_drifts),
        repositories_added=tuple(sorted(set(new_repos) - set(old_repos))),
        repositories_removed=tuple(sorted(set(old_repos) - set(new_repos))),
        containers_added=tuple(sorted(new_containers - old_containers)),
        containers_removed=tuple(sorted(old_containers - new_containers)),
        artifact_drifts=tuple(artifact_drifts),
    )
