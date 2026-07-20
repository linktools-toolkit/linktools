#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageFeatures: the capability surface a Storage declares.

Callers (and the RuntimeBuilder capability gate) branch on these values rather
than on concrete Storage/backend types or ``isinstance`` checks. Two scopes are
explicit enums rather than booleans so the builder can distinguish "no support"
from "process-local" from "distributed" -- a multi-worker Job or multi-process
Swarm requires the distributed end of the range, not merely a truthy flag.

StorageFeatures replaces StorageCapabilities: transaction/coordination are now
scopes (none/process_local/database|distributed), and streaming_blobs /
fencing are first-class so the capability gate can require them directly.
"""

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


# Coordination note: the in-repo reference Storage instances (FilesystemStorage and
# SqlAlchemyStorage) both ship the process-local ProcessLocalLeaseCoordinator,
# so both declare CoordinationScope.PROCESS_LOCAL. DISTRIBUTED coordination (a
# real cross-process lease backend -- Redis/etcd/a shared DB lease table) is a
# downstream concern: a deployment that needs multi-worker Jobs or multi-process
# Swarms injects a distributed LeaseCoordinator and declares DISTRIBUTED on its
# own StorageFeatures. The build-time capability gate
# (linktools.ai.run.requirements.enforce_storage_capability_gate) compares these
# scopes against a declared RuntimeRequirements and refuses a shortfall at
# Runtime.build time; subsystems opt into enforcement by passing requirements.


@dataclass(frozen=True, slots=True)
class StorageFeatures:
    transactions: TransactionScope
    coordination: CoordinationScope
    optimistic_concurrency: bool
    append_only_events: bool
    leasing: bool
    fencing: bool
    idempotency: bool
    streaming_blobs: bool
    full_text_search: bool
    semantic_search: bool
    multi_process_swarm: bool


FILE_STORAGE_FEATURES = StorageFeatures(
    transactions=TransactionScope.PROCESS_LOCAL,
    coordination=CoordinationScope.PROCESS_LOCAL,
    optimistic_concurrency=True,
    append_only_events=True,
    leasing=True,
    fencing=True,
    idempotency=True,
    streaming_blobs=True,
    full_text_search=False,
    semantic_search=False,
    multi_process_swarm=False,
)

SQLALCHEMY_STORAGE_FEATURES = StorageFeatures(
    transactions=TransactionScope.DATABASE,
    coordination=CoordinationScope.PROCESS_LOCAL,
    optimistic_concurrency=True,
    append_only_events=True,
    leasing=True,
    fencing=True,
    idempotency=True,
    streaming_blobs=True,
    full_text_search=True,
    semantic_search=False,
    multi_process_swarm=False,
)


__all__: "list[str]" = [
    "CoordinationScope",
    "FILE_STORAGE_FEATURES",
    "SQLALCHEMY_STORAGE_FEATURES",
    "StorageFeatures",
    "TransactionScope",
]
