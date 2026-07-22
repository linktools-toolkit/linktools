#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageFeatures: the capability surface a Storage declares, at component
granularity.

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
now cross-checks each declared component against the wired store, so a
declared-but-unwired component fails fast at build time."""

from dataclasses import dataclass
from enum import Enum


class TransactionScope(str, Enum):
    NONE = "none"
    PROCESS_LOCAL = "process_local"
    DATABASE = "database"


class CoordinationScope(str, Enum):
    NONE = "none"
    PROCESS_LOCAL = "process_local"
    DISTRIBUTED = "distributed"


class StorageComponent(str, Enum):
    """The store components a Storage may group into one transaction or offer
    optimistic concurrency for. Used by ``transactional_components`` and
    ``optimistic_concurrency`` so capabilities are declared per-store, not as a
    single global flag."""

    ASSETS = "assets"
    ARTIFACT_RECORDS = "artifact_records"
    RUNS = "runs"
    SESSIONS = "sessions"
    EVENTS = "events"
    APPROVALS = "approvals"
    CHECKPOINTS = "checkpoints"
    JOBS = "jobs"


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


# Coordination note: the in-repo reference Storage instances (FilesystemStorage
# and SqlAlchemyStorage) both ship the process-local ProcessLocalLeaseCoordinator,
# so both declare CoordinationScope.PROCESS_LOCAL. DISTRIBUTED coordination (a
# real cross-process lease backend -- Redis/etcd/a shared DB lease table) is a
# downstream concern: a deployment that needs multi-worker Jobs or multi-process
# Swarms injects a distributed LeaseCoordinator and declares DISTRIBUTED on its
# own StorageFeatures. The build-time capability gate compares these scopes
# against a declared RuntimeRequirements and refuses a shortfall at Runtime.build
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
)


__all__: "list[str]" = [
    "CoordinationScope",
    "FILE_STORAGE_FEATURES",
    "SQLALCHEMY_STORAGE_FEATURES",
    "StorageComponent",
    "StorageFeatures",
    "TransactionScope",
]
