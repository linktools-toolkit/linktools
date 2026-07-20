#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the public storage testkit Contracts against the from-scratch external
adapter in this package. This is the wheel-only conformance suite: it imports
ONLY the installed ``linktools.ai.storage.testing`` Contracts + this package's
own adapter (a sibling module) -- no source-tree, no in-repo relative path.

A downstream adapter package replaces ``adapter.py`` with its own backend and
re-runs this same suite in its CI."""

from linktools.ai.storage.testing import (  # installed wheel public surface
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
    LeaseCoordinatorContract,
)

from .adapter import (  # sibling module within THIS package
    InMemoryArtifactBlobStore,
    InMemoryArtifactRecordStore,
    InMemoryLeaseCoordinator,
)


class TestExternalBlobStoreConformance(ArtifactBlobStoreContract):
    def blob_store(self):
        return InMemoryArtifactBlobStore()


class TestExternalRecordStoreConformance(ArtifactRecordStoreContract):
    def record_store(self):
        return InMemoryArtifactRecordStore()


class TestExternalLeaseCoordinatorConformance(LeaseCoordinatorContract):
    def coordinator(self):
        return InMemoryLeaseCoordinator()
