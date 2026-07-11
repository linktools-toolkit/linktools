#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment Lock model (Spec Part VI section 39-40).

Records what a deployment's inputs and generated outputs *are*, for
reproducibility and drift detection -- never a runtime-state snapshot, and
never anything secret: no config plaintext, no passwords/tokens, no env
values, no full Compose/Manifest content, no user names, no host IP, no
absolute user_data path, no RUNNING_CONTAINERS.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class LockedRepository:
    url: str
    type: str  # "git" | "local"
    branch: "str | None"
    revision: "str | None"
    manifest_sha256: "str | None"
    manifest_version: "str | None"
    definitions_sha256: "str | None" = None  # local repos only (section 41)
    reproducible: bool = True


@dataclass(frozen=True)
class LockedContainer:
    name: str
    repository_url: "str | None"
    services: "tuple[str, ...]"


@dataclass(frozen=True)
class LockedArtifact:
    kind: str
    sha256: str


@dataclass(frozen=True)
class DeploymentLock:
    schema_version: int
    project: str
    linktools_cntr: str
    repositories: "tuple[LockedRepository, ...]"
    containers: "tuple[LockedContainer, ...]"
    artifacts: "dict[str, LockedArtifact]" = field(default_factory=dict)
