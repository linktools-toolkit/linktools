#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage composition contracts."""

from ._object_cleanup import ObjectCleanupResult
from ._plan import RuntimeDomain, RuntimeStoragePlan, runtime_domain_uses_object_store
from ._snapshot import (
    SnapshotExclusiveGuard,
    SnapshotLimits,
    snapshot_object_ref_payload,
    snapshot_object_ref_from_payload,
)
from ._input_capture import input_capture_key, input_capture_expiry_key, input_capture_object_dependency, iter_input_capture_dependencies
from ._root import RuntimeStorage
from ._conversation_persistence import ConversationPersistence
from ._checkpoint_persistence import CheckpointPersistence
from ._trace_persistence import TracePersistence
from ._contracts import ArtifactRecord, ArtifactRepositories, BudgetRepository

__all__ = [
    "ConversationPersistence",
    "CheckpointPersistence",
    "TracePersistence",
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
    "RuntimeStorage",
    "ArtifactRecord",
    "ArtifactRepositories",
    "RuntimeStoragePlan",
    "SnapshotLimits",
    "runtime_domain_uses_object_store",
]
