#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""external_adapter: a from-scratch, in-memory storage adapter that implements
the public Store Protocols and drives the full Runtime chain (run -> approval
-> resume -> artifact -> job) through the public surface ALONE.

This is the strong-form evidence: a wheel-only external
adapter. The package imports ONLY public ``linktools.ai.*`` paths -- never
``linktools.ai.runtime.builder`` and never the in-repo reference backends
(``storage.filesystem`` / ``storage.sqlalchemy`` / ``storage.coordination``).
The AST import guard in ``tests/test_wheel_isolation.py`` enforces this
mechanically; the connected-chain E2E in ``tests/test_runtime_e2e.py`` is the
behavioral proof.

The submodules are organized by domain per 's recommended layout:
``storage`` (the full storage surface), ``conformance_adapter`` (the smaller
blob/record/lease adapter the conformance testkit runs against)."""

from .commit import InMemoryRunCommitCoordinator
from .conformance_adapter import (
    InMemoryArtifactBlobStore,
    InMemoryArtifactRecordStore,
    InMemoryLeaseCoordinator,
)
from .storage import (
    InMemoryApprovalStore,
    InMemoryCheckpointStore,
    InMemoryEventStore,
    InMemoryExternalStorage,
    InMemoryIdempotencyStore,
    InMemoryJobStore,
    InMemoryMemoryStore,
    InMemoryRunDefinitionStore,
    InMemoryRunStore,
    InMemorySessionStore,
    InMemorySwarmStore,
    build_in_memory_external_storage,
)

__all__: "list[str]" = [
    "InMemoryApprovalStore",
    "InMemoryArtifactBlobStore",
    "InMemoryArtifactRecordStore",
    "InMemoryCheckpointStore",
    "InMemoryEventStore",
    "InMemoryExternalStorage",
    "InMemoryIdempotencyStore",
    "InMemoryJobStore",
    "InMemoryLeaseCoordinator",
    "InMemoryMemoryStore",
    "InMemoryRunCommitCoordinator",
    "InMemoryRunDefinitionStore",
    "InMemoryRunStore",
    "InMemorySessionStore",
    "InMemorySwarmStore",
    "build_in_memory_external_storage",
]
