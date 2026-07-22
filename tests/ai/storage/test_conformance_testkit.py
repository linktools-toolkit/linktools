#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the reusable storage conformance testkit against the in-repo reference
backends. This proves the testkit is importable and that the reference
backends satisfy the Protocol contracts a downstream adapter would be checked
against."""

import pytest

from linktools.ai.artifact.models import ArtifactIntegrityError
from linktools.ai.testing import (
    ArtifactBlobStoreContract,
    ArtifactRecordStoreContract,
)


from linktools.ai.testing import LeaseCoordinatorContract  # noqa: E402


class TestArtifactBlobStoreConformance(ArtifactBlobStoreContract):
    @pytest.fixture(autouse=True)
    def _cache_tmp_path(self, tmp_path) -> None:
        # The contract's blob_store() is no-arg; cache the per-test tmp_path so
        # each call returns a fresh, empty Filesystem store rooted here.
        self._tmp_path = tmp_path

    def blob_store(self):
        from linktools.ai.storage.filesystem.artifact import (
            FilesystemArtifactBlobStore,
        )

        return FilesystemArtifactBlobStore(
            blobs_root=self._tmp_path / "blobs"
        )

    # The contract references the integrity error; expose it for the mixin's
    # pytest.raises.
    ArtifactIntegrityError = ArtifactIntegrityError


class TestArtifactRecordStoreConformance(ArtifactRecordStoreContract):
    @pytest.fixture(autouse=True)
    def _cache_tmp_path(self, tmp_path) -> None:
        self._tmp_path = tmp_path

    def record_store(self):
        from linktools.ai.storage.filesystem.artifact import (
            FilesystemArtifactRecordStore,
        )

        return FilesystemArtifactRecordStore(
            records_root=self._tmp_path / "records"
        )


class TestLeaseCoordinatorConformance(LeaseCoordinatorContract):
    def coordinator(self):
        from linktools.ai.storage.coordination.process_local import (
            ProcessLocalLeaseCoordinator,
        )

        return ProcessLocalLeaseCoordinator()
