#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime storage composition contracts."""

from ._durability import CommitObservation, DurableCommitState, run_durable_commit, await_owned_task
from ._capability_contracts import CapabilityPersistence, CapabilityWriter
from ._archive_read import ArchiveNodeSlot, ArchiveNodeRead, ArchiveFrontierReader
from ._store import (StateStore, StateTransaction, StateStorageGroup, StateGroupTransaction, GroupRecordReader,
                     StoredRecord, StoredFact, StoredOperation, RecordReplacement, RecordQuery)
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
from ._runtime_state import RuntimeState, RuntimeStorageStatus
from ._snapshot_operations import RuntimeSnapshotOperations
from ._conversation_persistence import ConversationPersistence
from ._checkpoint_persistence import CheckpointPersistence
from ._trace_persistence import TracePersistence
from ._contracts import (ArtifactRecord, ArtifactRepositories, BudgetRepository,
                         ExecutionRepositories, TaskRepositories, EvaluationRepositories)

from ._capability_marker import ensure_capability_partition_marker, has_capability_partition_marker
from ._resource_scope import reject_filesystem_capability_partitions

__all__ = [
    "reject_filesystem_capability_partitions",
    "ensure_capability_partition_marker",
    "has_capability_partition_marker",
    "CommitObservation", "DurableCommitState", "run_durable_commit", "await_owned_task",
    "CapabilityPersistence", "CapabilityWriter",
    "ArchiveNodeSlot", "ArchiveNodeRead", "ArchiveFrontierReader",
    "StateStore", "StateTransaction", "StateStorageGroup", "StateGroupTransaction",
    "GroupRecordReader",
    "StoredRecord", "StoredFact", "StoredOperation", "RecordReplacement", "RecordQuery",
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
    "RuntimeState",
    "RuntimeStorageStatus",
    "RuntimeSnapshotOperations",
    "ExecutionRepositories",
    "TaskRepositories",
    "EvaluationRepositories",
    "ArtifactRecord",
    "ArtifactRepositories",
    "RuntimeStoragePlan",
    "SnapshotLimits",
    "runtime_domain_uses_object_store",
]
