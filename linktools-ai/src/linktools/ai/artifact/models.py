#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact domain models.

An artifact is immutable content -- a model request, a run output, a context
render, an eval case. The content is a content-addressed blob (deduplicated by
sha256); each production event is its own :class:`ArtifactRecord` (a UUID id)
carrying the provenance. So the same bytes share one blob but each put gets a
distinct record/lineage -- :class:`ArtifactRef`.id is the record id (UUID),
and ``sha256`` is the blob id.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A handle to an artifact. ``id`` is the ArtifactRecord id (a UUID) -- the
    lineage handle a caller carries. ``sha256`` is the content blob id, used to
    dedupe identical bytes. Splitting them lets two puts of the same content
    share one blob while each keeps its own record/lineage."""

    id: str
    sha256: str
    media_type: str
    size: int


@dataclass(frozen=True, slots=True)
class ArtifactBlob:
    """The content half of an artifact: immutable bytes keyed by sha256, shared
    by every ArtifactRecord that sealed the same content. Has no tenant/lineage
    -- those live on the per-write ArtifactRecord."""

    sha256: str
    media_type: str
    size: int
    path: str


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
    "ArtifactBlob",
    "ArtifactRecord",
    "ArtifactIntegrityError",
    "ResourceSnapshotRef",
]
