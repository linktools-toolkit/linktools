#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Artifact domain: immutable, content-addressed run products over AssetStore.

Public types for downstream and for the task / evaluation domains.
"""

from .models import ArtifactIntegrityError, ArtifactRecord, ArtifactRef, ResourceSnapshotRef
from .store import ArtifactStore

__all__: "list[str]" = [
    "ArtifactRef",
    "ArtifactRecord",
    "ArtifactStore",
    "ArtifactIntegrityError",
    "ResourceSnapshotRef",
]
