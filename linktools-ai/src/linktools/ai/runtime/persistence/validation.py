#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage feature-consistency validation.

The single canonical check that a Storage's declared :class:`StorageFeatures`
match its WIRED objects. Called from ``Storage.__post_init__`` so a
self-inconsistent Storage cannot be constructed at all, and re-used by
``RuntimeBuilder`` (which imports this function) so the build-time gate and
the construction-time gate are the SAME check -- two lists cannot drift.

A capability declared on features but not backed by a wired object (or vice
versa) is a fail-closed condition: the Storage / runtime build fails fast
rather than silently degrading into a capability it cannot actually deliver.
There is deliberately NO ``getattr(..., default)`` skip on the ArtifactStore:
a real ArtifactStore exposes ``coordination_scope`` / ``supports_streaming`` /
``record_store`` as public properties, so a stand-in missing them is a
misconfigured Storage that must be rejected."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...errors import StorageCapabilityDeclarationError, StorageRequirementsNotMetError
from ...storage.features import CoordinationScope, StorageComponent, TransactionScope
from .protocols import ArtifactStoreCapabilities, TransactionManagerCapabilities

if TYPE_CHECKING:
    from .facade import Storage


def validate_storage_feature_consistency(storage: "Storage") -> None:
    """Raise if ``storage.features`` disagrees with the objects actually wired
    onto ``storage``. Runs at construction; RuntimeBuilder calls the same
    function so the build-time gate cannot diverge."""
    f = storage.features

    from .features import StorageFeatures
    from .facade import storage_component_map
    from .protocols import ArtifactStoreCapabilities, TransactionManagerCapabilities

    # transaction_scope vs the wired transaction manager.
    if f.transaction_scope is not TransactionScope.NONE:
        manager = storage.transaction_manager
        if manager is None:
            raise StorageRequirementsNotMetError(
                f"Storage declares transaction_scope={f.transaction_scope.value!r} "
                "but its transaction manager is None"
            )
        if not isinstance(manager, TransactionManagerCapabilities):
            raise StorageCapabilityDeclarationError(
                "wired transaction manager must expose scope"
            )
        if f.transaction_scope is TransactionScope.DATABASE and manager.scope is not TransactionScope.DATABASE:
            raise StorageRequirementsNotMetError(
                "Storage declares transaction_scope=DATABASE but its transaction "
                "manager is NoCrossStoreTransactions (cannot group stores)"
            )

    # coordination_scope / leasing / fencing vs the wired LeaseCoordinator.
    if f.coordination_scope is not CoordinationScope.NONE and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            f"Storage declares coordination_scope={f.coordination_scope.value!r} "
            "but its LeaseCoordinator is None"
        )
    if f.leasing and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            "Storage declares leasing=True but its LeaseCoordinator is None"
        )
    if f.fencing and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            "Storage declares fencing=True but its LeaseCoordinator is None "
            "(fencing tokens are minted by the coordinator)"
        )

    # streaming_artifacts / artifact_coordination_scope vs the wired
    # ArtifactStore. Fail CLOSED: a real ArtifactStore exposes the public
    # capability properties; a stand-in missing them is rejected, not skipped.
    if f.streaming_artifacts and storage.artifacts is None:
        raise StorageRequirementsNotMetError(
            "Storage declares streaming_artifacts=True but its ArtifactStore "
            "(storage.artifacts) is None"
        )
    if storage.artifacts is not None:
        artifacts = storage.artifacts
        if not f.streaming_artifacts:
            raise StorageRequirementsNotMetError(
                "Storage wires an ArtifactStore but declares "
                "streaming_artifacts=False -- the flag must agree with the wired store"
            )
        if not isinstance(artifacts, ArtifactStoreCapabilities):
            raise StorageCapabilityDeclarationError(
                "wired ArtifactStore must expose complete capabilities"
            )
        wired_scope = artifacts.coordination_scope
        if wired_scope != f.artifact_coordination_scope:
            raise StorageRequirementsNotMetError(
                f"Storage declares artifact_coordination_scope="
                f"{f.artifact_coordination_scope.value!r} but its wired "
                f"ArtifactStore coordinator scope is {wired_scope.value!r}"
            )

    if not isinstance(storage.transaction_manager, TransactionManagerCapabilities):
        raise StorageCapabilityDeclarationError(
            "wired transaction manager must expose scope"
        )
    if storage.artifacts is not None and not isinstance(
        storage.artifacts, ArtifactStoreCapabilities
    ):
        raise StorageCapabilityDeclarationError(
            "wired ArtifactStore must expose complete capabilities"
        )

    wired = frozenset(
        component
        for component, store in storage_component_map(storage).items()
        if store is not None
    )
    missing = f.transactional_components - wired
    if missing:
        names = ", ".join(sorted(c.value for c in missing))
        raise StorageRequirementsNotMetError(
            f"Storage declares transactional_components that are not wired: {names}"
        )

    expected = StorageFeatures.from_components(
        transaction_manager=storage.transaction_manager,
        coordination=storage.coordination,
        artifacts=storage.artifacts,
        components=storage_component_map(storage),
    )
    if f != expected:
        raise StorageCapabilityDeclarationError(
            "Storage.features does not match capabilities derived from wired objects"
        )

    # Each declared transactional component must be backed by a real wired
    # store. Use the same component map __post_init__ derived features from,
    # so the declared set and the wired set come from one source of truth.
    wired = frozenset(
        component
        for component, store in storage_component_map(storage).items()
        if store is not None
    )
    missing = f.transactional_components - wired
    if missing:
        names = ", ".join(sorted(c.value for c in missing))
        raise StorageRequirementsNotMetError(
            f"Storage declares transactional_components that are not wired: {names}"
        )
    if (
        StorageComponent.JOBS in f.transactional_components
        and storage.jobs is None
    ):
        raise StorageRequirementsNotMetError(
            "Storage declares JOBS transactional but storage.jobs is None"
        )


__all__: "list[str]" = ["validate_storage_feature_consistency"]
