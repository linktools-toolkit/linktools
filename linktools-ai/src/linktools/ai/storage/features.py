#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage capability vocabulary: the enums a downstream Storage declares as
its capability surface. Pure standard-library (no domain imports), so this
module stays in the storage kernel.

The runtime-composition surface that BUILDS ON these enums (``StorageFeatures``
dataclass + the ``FILE_STORAGE_FEATURES`` / ``SQLALCHEMY_STORAGE_FEATURES``
presets) lives at ``linktools.ai.runtime.persistence.features`` -- it is
runtime-shaped (a frozen declaration of what THIS Storage provides), not a
storage-kernel primitive."""

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
    ``optimistic_concurrency`` so capabilities are declared per-store, not as
    a single global flag."""

    ASSETS = "assets"
    ARTIFACT_RECORDS = "artifact_records"
    RUNS = "runs"
    SESSIONS = "sessions"
    EVENTS = "events"
    APPROVALS = "approvals"
    CHECKPOINTS = "checkpoints"
    JOBS = "jobs"


__all__: "list[str]" = [
    "CoordinationScope",
    "StorageComponent",
    "TransactionScope",
]
