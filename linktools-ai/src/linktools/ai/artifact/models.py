#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact domain models.

An artifact is an immutable, content-addressed blob -- a model request, a run
output, a context render, an eval case -- tracked by an :class:`ArtifactRef`
plus a provenance :class:`ArtifactRecord`. The id IS the content's sha256, so
the same bytes always resolve to the same artifact.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    id: str
    sha256: str
    media_type: str
    size: int


@dataclass(frozen=True, slots=True)
class ResourceSnapshotRef:
    """A resource pinned into an immutable artifact: the original path/version/
    etag plus the content-addressed artifact id and sha256. Lives in the
    artifact domain (the lowest layer both ``task`` and ``evaluation`` depend
    on) so neither has to reach across the other to reference it."""

    path: str
    version: int
    etag: str
    artifact_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    ref: ArtifactRef
    tenant_id: str
    created_by_job_id: "str | None"
    created_by_task_id: "str | None"
    created_by_attempt_id: "str | None"
    parent_artifact_ids: "tuple[str, ...]"
    created_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


class ArtifactIntegrityError(Exception):
    """Raised when stored artifact content does not match its sha256."""


__all__: "list[str]" = [
    "ArtifactRef",
    "ArtifactRecord",
    "ArtifactIntegrityError",
    "ResourceSnapshotRef",
]
