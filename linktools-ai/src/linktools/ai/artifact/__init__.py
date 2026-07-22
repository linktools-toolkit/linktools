#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact domain: immutable, content-addressed run products over the
ArtifactBlobStore / ArtifactRecordStore Protocols (decoupled from any specific
backend -- asset store, filesystem, or external object store).

Public types for downstream and for the task / evaluation domains.
"""

from .models import (
    ANONYMOUS_PROVENANCE,
    ArtifactBlobNotFoundError,
    ArtifactBufferedSizeLimitError,
    ArtifactIntegrityError,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRef,
    ArtifactStagingError,
    AssetSnapshotRef,
)
from .store import ArtifactStore

__all__: "list[str]" = [
    "ArtifactRef",
    "ArtifactProvenance",
    "ArtifactRecord",
    "ArtifactStore",
    "ArtifactBlobNotFoundError",
    "ArtifactIntegrityError",
    "ArtifactBufferedSizeLimitError",
    "ArtifactStagingError",
    "ANONYMOUS_PROVENANCE",
    "AssetSnapshotRef",
]
