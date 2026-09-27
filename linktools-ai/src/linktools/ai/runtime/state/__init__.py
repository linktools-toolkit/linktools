#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage composition contracts."""

from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
    runtime_domain_uses_object_store,
)
from ._snapshot import SnapshotExclusiveGuard, SnapshotLimits
from .task_capability_capture import (
    TASK_CAPABILITY_CAPTURE_FORMAT_VERSION,
    read_task_capability_capture_declarations,
    task_declaration_identity,
    task_expander_declaration_identity,
)
from ._root import RuntimeStorage
from ._contracts import ArtifactRecord, ArtifactRepositories

__all__ = [
    "SnapshotExclusiveGuard",
    "RuntimeDomain",
    "RuntimeRetentionMode",
    "RuntimeStorage",
    "ArtifactRecord",
    "ArtifactRepositories",
    "RuntimeStoragePlan",
    "RuntimeStorageRoute",
    "SnapshotLimits",
    "TASK_CAPABILITY_CAPTURE_FORMAT_VERSION",
    "read_task_capability_capture_declarations",
    "task_declaration_identity",
    "task_expander_declaration_identity",
    "runtime_domain_uses_object_store",
]
