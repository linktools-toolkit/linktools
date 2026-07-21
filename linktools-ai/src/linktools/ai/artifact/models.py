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
from typing import TypeAlias

# A JSON value: scalar | dict[str, JsonValue] | list[JsonValue]. The recursive
# alias is a string so the self-reference resolves lazily (no runtime eval).
JsonValue: "TypeAlias" = (
    "str | int | float | bool | None | dict[str, JsonValue] | list[JsonValue]"
)


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
class ArtifactProvenance:
    """Generic provenance carried by an :class:`ArtifactRecord`: who produced
    it (``producer_kind`` + ``producer_id``), the optional run/session that
    framed the production, any parent artifacts it was derived from, and a
    free-form metadata bag. Keeping provenance in one value object lets the
    artifact domain model lineage WITHOUT importing the jobs domain -- a job
    attempt, an eval, a CLI import, etc. each construct an ``ArtifactProvenance``
    through the same surface."""

    producer_kind: str
    producer_id: str
    run_id: "str | None" = None
    session_id: "str | None" = None
    parent_artifact_ids: "tuple[str, ...]" = ()
    metadata: "Mapping[str, JsonValue]" = field(default_factory=dict)


# A shared anonymous provenance for callers that have no real producer context
# (best-effort seals, test fixtures). ArtifactStore.put/put_stream require an
# explicit provenance; this constant is the no-op producer.
ANONYMOUS_PROVENANCE = ArtifactProvenance(producer_kind="anonymous", producer_id="")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    ref: ArtifactRef
    tenant_id: str
    provenance: ArtifactProvenance
    created_at: datetime


class ArtifactBlobNotFoundError(Exception):
    """Raised when a content-addressed blob is absent -- the requested digest
    has no blob behind it (e.g. an orphaned record whose blob was swept, or a
    digest that was never written). Distinct from
    :class:`ArtifactIntegrityError` (which means the blob EXISTS but its bytes
    do not hash back to the recorded sha256): a caller can tell "does not
    exist" apart from "exists but is corrupt." Plan §4.1 names this class as
    the unified signal for the missing case across every blob backend."""


class ArtifactIntegrityError(Exception):
    """Raised when stored artifact content does not match its sha256, or its
    size does not match the recorded size -- i.e. the blob EXISTS but is
    corrupt/tampered. A MISSING blob raises :class:`ArtifactBlobNotFoundError`
    instead."""


class ArtifactBufferedSizeLimitError(Exception):
    """Raised when the buffered (whole-bytes) API is used for content that
    exceeds the bounded-memory threshold. Callers must use the streaming API
    (``put_stream`` / ``open_stream``) for content at or above the limit, so the
    facade never materializes a whole artifact into a single ``bytes``."""


class ArtifactStagingError(Exception):
    """Raised when the streaming-put staging path cannot complete its temporary
    file I/O -- most commonly disk-full (``OSError`` / ``ENOSPC``) on the
    staging spool, but also any other I/O failure while hashing + spilling the
    source. Wrapping the raw ``OSError`` gives callers a single domain error to
    handle instead of a platform-specific errno. Plan §4.2 line 368."""


__all__: "list[str]" = [
    "ArtifactRef",
    "ArtifactBlob",
    "ArtifactProvenance",
    "ArtifactRecord",
    "ArtifactBlobNotFoundError",
    "ArtifactIntegrityError",
    "ArtifactBufferedSizeLimitError",
    "ArtifactStagingError",
    "ResourceSnapshotRef",
]
