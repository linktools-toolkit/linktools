#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the public storage testkit Contracts against the from-scratch external
adapter in this package. This is the wheel-only conformance suite for adapter
CODE: ``adapter.py`` imports ONLY the installed ``linktools.ai.*`` public
surface -- no source-tree, no in-repo relative path. The testkit itself
(``linktools.ai.testing``, from the separate ``linktools-ai-testing`` wheel) is
test-support code that ships alongside this package rather than inside the
core wheel.

A downstream adapter package replaces ``adapter.py`` with its own backend and
re-runs this same suite in its CI."""

from linktools.ai.testing import (
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
