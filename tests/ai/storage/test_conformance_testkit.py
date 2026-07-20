#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the reusable storage conformance testkit against the in-repo reference
backends. This proves the testkit is importable and that the reference
backends satisfy the Protocol contracts a downstream adapter would be checked
against."""

import hashlib

import pytest

from linktools.ai.asset.memory import MemoryAssetBackend
from linktools.ai.asset.path import AssetPath
from linktools.ai.asset.store import AssetStore
from linktools.ai.artifact.models import ArtifactIntegrityError
from linktools.ai.storage.testing import (
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
)


from linktools.ai.storage.testing import LeaseCoordinatorContract  # noqa: E402


class TestArtifactBlobStoreConformance(ArtifactBlobStoreContract):
    def blob_store(self):
        from linktools.ai.storage.artifact_backends import (
            AssetBackedArtifactBlobStore,
        )

        assets = AssetStore(primary=MemoryAssetBackend())
        return AssetBackedArtifactBlobStore(
            assets, blobs_root=AssetPath("/artifacts/blobs/sha256")
        )

    # The contract references the integrity error; expose it for the mixin's
    # pytest.raises.
    ArtifactIntegrityError = ArtifactIntegrityError


class TestArtifactRecordStoreConformance(ArtifactRecordStoreContract):
    def record_store(self):
        from linktools.ai.storage.artifact_backends import (
            AssetBackedArtifactRecordStore,
        )

        assets = AssetStore(primary=MemoryAssetBackend())
        return AssetBackedArtifactRecordStore(
            assets, records_root=AssetPath("/artifacts/records")
        )


class TestLeaseCoordinatorConformance(LeaseCoordinatorContract):
    def coordinator(self):
        from linktools.ai.storage.coordination.process_local import (
            ProcessLocalLeaseCoordinator,
        )

        return ProcessLocalLeaseCoordinator()
