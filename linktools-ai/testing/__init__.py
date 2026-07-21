#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable storage conformance testkit.

This package lives OUTSIDE ``linktools-ai/src/`` -- it is test-support code,
not library code, so it is never packaged into the ``linktools-ai`` wheel.
Downstream adapters (an external object store, a different SQL dialect, a
distributed lease coordinator) subclass these contract mixins, supply a
``store``/``coordinator`` fixture built from their own backend, and get the
same conformance checks the in-repo reference backends must pass. The testkit
depends on pytest.

Each contract is the observable business semantics a Protocol promises, not the
implementation details behind it. A backend that fails a contract is not
conformant; the RuntimeBuilder capability gate can then refuse to wire it.

Importable as ``testing`` by any suite that puts ``linktools-ai/`` on
``sys.path`` (see ``tests/conftest.py`` and ``linktools-ai/conformance/conftest.py``)."""

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
