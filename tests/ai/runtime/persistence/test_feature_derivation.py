#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage feature derivation + fail-closed capability gate.

Features are DERIVED from the real wired objects (transaction manager,
coordinator, artifact store, each component store's ``capabilities``); callers
cannot inject ``leasing=True`` style bool declarations. A wired ArtifactStore
missing its public capability properties fails construction."""

from __future__ import annotations

import pytest

from linktools.ai.errors import StorageCapabilityDeclarationError
from linktools.ai.runtime.persistence.facade import (
    FilesystemStorage,
    storage_component_map,
)
from linktools.ai.runtime.persistence.features import StorageFeatures
from linktools.ai.runtime.persistence.transaction import NoCrossStoreTransactions
from linktools.ai.storage.features import (
    CoordinationScope,
    StorageComponent,
    TransactionScope,
)


def test_artifact_records_in_component_map(tmp_path):
    """ARTIFACT_RECORDS is in the wired component map (surfaced via
    ArtifactStore.record_store)."""
    storage = FilesystemStorage(root=tmp_path)
    components = storage_component_map(storage)
    assert StorageComponent.ARTIFACT_RECORDS in components
    assert components[StorageComponent.ARTIFACT_RECORDS] is storage.artifacts.record_store


def test_features_derived_from_real_objects_no_caller_bools(tmp_path):
    """Features come from the wired objects, not caller-supplied bools.
    FilesystemStorage wires a PROCESS_LOCAL coordinator + an ArtifactStore, so
    coordination_scope=PROCESS_LOCAL, leasing/fencing=True, streaming=True."""
    storage = FilesystemStorage(root=tmp_path)
    f = storage.features
    assert f.transaction_scope is TransactionScope.NONE  # Filesystem: no cross-store txn
    assert f.coordination_scope is CoordinationScope.PROCESS_LOCAL
    assert f.artifact_coordination_scope is CoordinationScope.PROCESS_LOCAL
    assert f.leasing is True
    assert f.fencing is True
    assert f.streaming_artifacts is True


def test_coordination_none_derives_none_and_false(tmp_path):
    """coordination=None derives coordination_scope=NONE, leasing=False,
    fencing=False. Built from a minimal features object (no coordinator)."""
    f = StorageFeatures.from_components(
        transaction_manager=NoCrossStoreTransactions(),
        coordination=None,
        artifacts=None,
        components={},
    )
    assert f.coordination_scope is CoordinationScope.NONE
    assert f.leasing is False
    assert f.fencing is False
    assert f.streaming_artifacts is False


def test_coordinator_without_fencing_derives_fencing_false():
    """A coordinator that declares supports_fencing=False derives fencing=False
    even though it is wired."""
    class _NoFencingCoord:
        scope = CoordinationScope.PROCESS_LOCAL
        supports_leasing = True
        supports_fencing = False

        async def acquire(self, **kwargs): ...
        async def renew(self, **kwargs): ...
        async def release(self, **kwargs): ...

    f = StorageFeatures.from_components(
        transaction_manager=NoCrossStoreTransactions(),
        coordination=_NoFencingCoord(),
        artifacts=None,
        components={},
    )
    assert f.leasing is True
    assert f.fencing is False


def test_storage_construction_rejects_artifactstore_missing_capability_property(tmp_path):
    """A wired ArtifactStore that does NOT expose the public capability
    properties fails construction (fail-closed), not silent skip."""
    from linktools.ai.runtime.persistence.validation import (
        validate_storage_feature_consistency,
    )

    storage = FilesystemStorage(root=tmp_path)

    class _BareArtifactStore:
        # Has coordination_scope + supports_streaming but NO record_store.
        coordination_scope = CoordinationScope.PROCESS_LOCAL
        supports_streaming = True

    # Swap the wired artifacts for a bare stand-in missing record_store; the
    # consistency gate must reject it rather than skip the check.
    object.__setattr__(storage, "artifacts", _BareArtifactStore())
    with pytest.raises(StorageCapabilityDeclarationError):
        validate_storage_feature_consistency(storage)


def test_runtimebuilder_reuses_same_validation_function():
    """RuntimeBuilder calls the SAME validation function
    Storage.__post_init__ uses (no duplicated gate). The builder's
    enforce_storage_feature_consistency is the canonical validator."""
    from linktools.ai.run.requirements import enforce_storage_feature_consistency
    from linktools.ai.runtime.persistence.validation import (
        validate_storage_feature_consistency,
    )

    # The builder entrypoint must delegate to the canonical validator (it does
    # not reimplement the checks).
    import inspect

    src = inspect.getsource(enforce_storage_feature_consistency)
    assert "validate_storage_feature_consistency" in src
    assert validate_storage_feature_consistency is not enforce_storage_feature_consistency
