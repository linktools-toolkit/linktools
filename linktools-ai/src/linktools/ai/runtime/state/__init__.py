#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage composition contracts."""

from ._object_cleanup import ObjectCleanupResult
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStoragePlan,
    RuntimeStorageRoute,
    runtime_domain_uses_object_store,
)
from ._snapshot import (
    SnapshotExclusiveGuard,
    SnapshotLimits,
    snapshot_object_ref_payload,
    snapshot_object_ref_from_payload,
)
from ._input_capture import input_capture_key, input_capture_expiry_key, input_capture_object_dependency, iter_input_capture_dependencies
from ._root import RuntimeStorage
from ._contracts import ArtifactRecord, ArtifactRepositories, BudgetRepository

__all__ = [
    "BudgetRepository",
    "ObjectCleanupResult",
    "input_capture_key",
    "input_capture_expiry_key",
    "input_capture_object_dependency",
    "iter_input_capture_dependencies",
    "snapshot_object_ref_payload",
    "snapshot_object_ref_from_payload",
    "SnapshotExclusiveGuard",
    "RuntimeDomain",
    "RuntimeRetentionMode",
    "RuntimeStorage",
    "ArtifactRecord",
    "ArtifactRepositories",
    "RuntimeStoragePlan",
    "RuntimeStorageRoute",
    "SnapshotLimits",
    "runtime_domain_uses_object_store",
]
