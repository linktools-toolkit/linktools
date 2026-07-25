#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence features: ``StorageFeatures`` (the capability surface a
Storage declares, at component granularity) + the two in-repo reference
presets (``FILE_STORAGE_FEATURES``, ``SQLALCHEMY_STORAGE_FEATURES``).

The capability enums themselves (``TransactionScope`` / ``CoordinationScope`` /
``StorageComponent``) live at ``linktools.ai.storage.features`` -- they are the
storage-kernel vocabulary a downstream Storage declares. ``StorageFeatures``
composes those enums into a frozen declaration of what THIS Storage provides,
so it lives here at the runtime layer (next to the Storage composition root).

A Storage declares which components it transactionally groups, which support
optimistic concurrency, and the scope of its transaction/coordination
providers. Callers (and the RuntimeBuilder capability gate) branch on these
values rather than on concrete Storage/backend types or ``isinstance`` checks.
Two scopes are explicit enums so the builder can distinguish "no support" from
"process-local" from "distributed" -- a multi-worker Job or multi-process
Swarm requires the distributed end of the range, not merely a truthy flag.

The component-level fields (``transactional_components`` /
``optimistic_concurrency``) replace the former global-bool declaration: a
single ``transactions=DATABASE`` flag could not express that, e.g., a backend
groups Runs+Events atomically but leaves Assets out. The consistency gate
(:func:`~linktools.ai.run.requirements.enforce_storage_feature_consistency`)
cross-checks each declared component against the wired store, so a
declared-but-unwired component fails fast at build time."""

from dataclasses import dataclass

from ...storage.features import CoordinationScope, StorageComponent, TransactionScope

_ALL_COMPONENTS: "frozenset[StorageComponent]" = frozenset(StorageComponent)


@dataclass(frozen=True, slots=True)
class StorageFeatures:
    transaction_scope: TransactionScope
    transactional_components: "frozenset[StorageComponent]"
    coordination_scope: CoordinationScope
    optimistic_concurrency: "frozenset[StorageComponent]"
    leasing: bool
    fencing: bool
    idempotency: bool
    streaming_artifacts: bool
    append_only_events: bool
    # The scope of the ArtifactStore's KeyedCoordinator (put/sweep mutual
    # exclusion) -- a SEPARATE capability from ``coordination_scope`` (the Job
    # Lease coordinator's scope). A Storage with no ArtifactStore wired
    # declares NONE here; one with an ArtifactStore declares whatever its
    # injected KeyedCoordinator's own ``.scope`` is.
    artifact_coordination_scope: CoordinationScope = CoordinationScope.NONE


# Coordination note: the in-repo reference Storage instances (FilesystemStorage
# and SqlAlchemyStorage) both ship the process-local ProcessLocalLeaseCoordinator,
# so both declare CoordinationScope.PROCESS_LOCAL. DISTRIBUTED coordination (a
# real cross-process lease backend -- Redis/etcd/a shared DB lease table) is a
# downstream concern: a deployment that needs multi-worker Jobs or multi-process
# Swarms injects a distributed LeaseCoordinator and declares DISTRIBUTED on its
# own StorageFeatures. The build-time capability gate compares these scopes
# against a declared RuntimeRequirements and refuses a shortfall at build_runtime
# time; subsystems opt into enforcement by passing requirements.


FILE_STORAGE_FEATURES = StorageFeatures(
    # NONE: each file store is independently durable (atomic writes), but there
    # is NO general cross-store transaction -- Storage.transaction() raises.
    transaction_scope=TransactionScope.NONE,
    transactional_components=frozenset(),
    coordination_scope=CoordinationScope.PROCESS_LOCAL,
    optimistic_concurrency=_ALL_COMPONENTS,
    leasing=True,
    fencing=True,
    idempotency=True,
    streaming_artifacts=True,
    append_only_events=True,
    # FilesystemStorage wires ArtifactStore with the default
    # InProcessKeyedCoordinator -> PROCESS_LOCAL.
    artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
)

SQLALCHEMY_STORAGE_FEATURES = StorageFeatures(
    # DATABASE: one AsyncSession + one transaction groups every store, so all
    # components commit or roll back together.
    transaction_scope=TransactionScope.DATABASE,
    transactional_components=_ALL_COMPONENTS,
    coordination_scope=CoordinationScope.PROCESS_LOCAL,
    optimistic_concurrency=_ALL_COMPONENTS,
    leasing=True,
    fencing=True,
    idempotency=True,
    streaming_artifacts=True,
    append_only_events=True,
    # SqlAlchemyStorage wires ArtifactStore with a FilesystemKeyedCoordinator
    # -> PROCESS_LOCAL (flock coordinates one filesystem, not distributed workers).
    artifact_coordination_scope=CoordinationScope.PROCESS_LOCAL,
)


__all__: "list[str]" = [
    "FILE_STORAGE_FEATURES",
    "SQLALCHEMY_STORAGE_FEATURES",
    "StorageFeatures",
]
