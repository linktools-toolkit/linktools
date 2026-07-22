#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable storage conformance testkit.

This package ships from an INDEPENDENT distribution (``linktools-ai-testing``)
-- it is test-support code, not library code, so it is never packaged into the
``linktools-ai`` wheel. Downstream adapters (an external object store, a
different SQL dialect, a distributed lease coordinator) subclass these contract
mixins, supply a ``store``/``coordinator`` fixture built from their own
backend, and get the same conformance checks the in-repo reference backends
must pass. The testkit depends on pytest.

Each contract is the observable business semantics a Protocol promises, not the
implementation details behind it. A backend that fails a contract is not
conformant; the RuntimeBuilder capability gate can then refuse to wire it.

Importable as ``linktools.ai.testing`` once the ``linktools-ai-testing`` wheel
is installed alongside ``linktools-ai``; in-repo suites resolve it the same way
via the ``linktools-ai-testing/src`` entry on ``sys.path`` (see
``tests/conftest.py``)."""

from .contracts import (
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
    AssetStoreContract,
    EventStoreContract,
    JobStoreContract,
    LeaseCoordinatorContract,
    StorageFeaturesContract,
    StorageTransactionManagerContract,
)

__all__: "list[str]" = [
    "ArtifactBlobStoreContract",
    "ArtifactRecordStoreContract",
    "AssetStoreContract",
    "EventStoreContract",
    "JobStoreContract",
    "LeaseCoordinatorContract",
    "StorageFeaturesContract",
    "StorageTransactionManagerContract",
]
