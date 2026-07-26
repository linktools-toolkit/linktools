#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence features: ``StorageFeatures`` (the capability surface a
Storage declares, at component granularity) + the two in-repo reference
static capability presets.

The capability enums themselves (``TransactionScope`` / ``CoordinationScope`` /
``StorageComponent``) live at ``linktools.ai.storage.features`` -- they are the
storage-kernel vocabulary a downstream Storage declares. ``StorageFeatures``
composes those enums into a frozen declaration of what THIS Storage provides,
so it lives here at the runtime layer (next to the Storage composition root).

A Storage declares which components it transactionally groups, which support
optimistic concurrency, and the scope of its transaction/coordination
providers. Callers (and the RuntimeBuilder capability gate) branch on these
values rather than on concrete Storage/backend types or ``isinstance`` checks.
Two scopes are explicit enums so the builder can distinguish "no support" from
"process-local" from "distributed" -- a multi-worker Job or multi-process
Swarm requires the distributed end of the range, not merely a truthy flag.

The component-level fields (``transactional_components`` /
``optimistic_concurrency``) replace the former global-bool declaration: a
single ``transactions=DATABASE`` flag could not express that, e.g., a backend
groups Runs+Events atomically but leaves Assets out. The consistency gate
(:func:`~linktools.ai.run.requirements.enforce_storage_feature_consistency`)
cross-checks each declared component against the wired store, so a
declared-but-unwired component fails fast at build time."""

from dataclasses import dataclass, field
from typing import Mapping

from ...storage.features import (
    ComponentCapabilities,
    CoordinationScope,
    StorageComponent,
    TransactionScope,
)
from ...errors import StorageCapabilityDeclarationError
from .protocols import (
    ArtifactStoreCapabilities,
    ComponentCapabilityProvider,
    LeaseCoordinatorCapabilities,
    TransactionManagerCapabilities,
)

def _capabilities_of(store: object) -> ComponentCapabilities:
    """Read a wired component's declared capabilities fail-closed."""
    if not isinstance(store, ComponentCapabilityProvider):
        raise StorageCapabilityDeclarationError(
            "every wired component store must expose ComponentCapabilities"
        )
    caps = store.capabilities
    if not isinstance(caps, ComponentCapabilities):
        raise StorageCapabilityDeclarationError(
            "wired component store capabilities has the wrong type"
        )
    return caps


@dataclass(frozen=True, slots=True)
class StorageFeatures:
    transaction_scope: TransactionScope
    transactional_components: "frozenset[StorageComponent]"
    coordination_scope: CoordinationScope
    optimistic_concurrency: "frozenset[StorageComponent]"
    leasing: bool
    fencing: bool
    idempotency: bool
    streaming_artifacts: bool
    append_only_events: bool
    # The scope of the ArtifactStore's KeyedCoordinator (put/sweep mutual
    # exclusion) -- a SEPARATE capability from ``coordination_scope`` (the Job
    # Lease coordinator's scope). A Storage with no ArtifactStore wired
    # declares NONE here; one with an ArtifactStore declares whatever its
    # injected KeyedCoordinator's own ``.scope`` is.
    artifact_coordination_scope: CoordinationScope = CoordinationScope.NONE

    @classmethod
    def from_components(
        cls,
        *,
        transaction_manager: "object | None",
        coordination: "object | None",
        artifacts: "object | None",
        components: "Mapping[StorageComponent, object | None]",
    ) -> "StorageFeatures":
        """Build features by inspecting the REAL wired objects -- the
        transaction manager, the lease coordinator, the artifact store, and
        each component store's ``capabilities``. Callers supply objects, never
        ``leasing=True`` / ``fencing=True`` style bool declarations, so a
        Storage cannot claim a capability none of its wired objects provide.

        - transaction_scope/coordination_scope/artifact_coordination_scope are
          read off the wired objects (NONE when the object is absent);
        - leasing/fencing are True iff a coordinator is wired AND it declares
          supports_leasing / supports_fencing;
        - streaming_artifacts is True iff an ArtifactStore is wired AND it
          declares supports_streaming;
        - transactional_components / optimistic_concurrency / idempotency /
          append_only_events are derived from each component store's
          ``capabilities`` (a component appears in a set iff its wired store
          declares the corresponding capability)."""
        if transaction_manager is None:
            raise StorageCapabilityDeclarationError("transaction_manager is required")
        if not isinstance(transaction_manager, TransactionManagerCapabilities):
            raise StorageCapabilityDeclarationError("transaction_manager must expose scope")
        transaction_scope = transaction_manager.scope
        if not isinstance(transaction_scope, TransactionScope):
            raise StorageCapabilityDeclarationError("transaction_manager.scope has invalid type")
        if coordination is None:
            coordination_scope = CoordinationScope.NONE
            leasing = fencing = False
        else:
            if not isinstance(coordination, LeaseCoordinatorCapabilities):
                raise StorageCapabilityDeclarationError("coordination must expose complete capabilities")
            coordination_scope = coordination.scope
            leasing = coordination.supports_leasing
            fencing = coordination.supports_fencing
            if not isinstance(coordination_scope, CoordinationScope) or not isinstance(leasing, bool) or not isinstance(fencing, bool):
                raise StorageCapabilityDeclarationError("invalid coordination capability declaration")
        if artifacts is None:
            artifact_coordination_scope = CoordinationScope.NONE
            streaming_artifacts = False
        else:
            if not isinstance(artifacts, ArtifactStoreCapabilities):
                raise StorageCapabilityDeclarationError("artifacts must expose complete capabilities")
            artifact_coordination_scope = artifacts.coordination_scope
            streaming_artifacts = artifacts.supports_streaming
            if not isinstance(artifact_coordination_scope, CoordinationScope) or not isinstance(streaming_artifacts, bool):
                raise StorageCapabilityDeclarationError("invalid artifact capability declaration")
        transactional: "set[StorageComponent]" = set()
        optimistic: "set[StorageComponent]" = set()
        idempotency_seen = False
        append_only_seen = False
        for component, store in components.items():
            if store is None:
                continue
            caps = _capabilities_of(store)
            if caps.transaction_participation:
                transactional.add(component)
            if caps.optimistic_concurrency:
                optimistic.add(component)
            if caps.idempotency:
                idempotency_seen = True
            if caps.append_only:
                append_only_seen = True
        return cls(
            transaction_scope=transaction_scope,
            transactional_components=frozenset(transactional),
            coordination_scope=coordination_scope,
            optimistic_concurrency=frozenset(optimistic),
            leasing=leasing,
            fencing=fencing,
            idempotency=idempotency_seen,
            streaming_artifacts=streaming_artifacts,
            append_only_events=append_only_seen,
            artifact_coordination_scope=artifact_coordination_scope,
        )


__all__: "list[str]" = [
    "ComponentCapabilities",
    "StorageFeatures",
]
