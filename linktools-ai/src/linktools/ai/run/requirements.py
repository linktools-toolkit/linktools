#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeRequirements: what a runtime topology NEEDS from storage, declared as
capability minimums. The RuntimeBuilder capability gate compares these against
:class:`~linktools.ai.storage.features.StorageFeatures` (what storage
DECLARES) and fails fast at build time on a shortfall -- never silently
degrading to a weaker scope.

This is the capability gate: it branches on capability scope/flag values,
never on ``isinstance`` against a concrete backend. A caller whose topology
needs, e.g., distributed coordination passes
``build_runtime(requirements=RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED))``
and the builder refuses a process-local Storage rather than letting it race.

Every field defaults to the most permissive value (no requirement), so a bare
``RuntimeRequirements()`` -- or passing ``None`` -- imposes no gate. Callers
opt INTO enforcement by raising the minimums their topology actually needs;
subsystems (a multi-worker JobRuntime, a multi-process Swarm) declare their
own requirements in their own wiring.

``RuntimeTopology`` is the lightweight declarative enum a caller hands the
build kernel when it does not want to spell out a full RuntimeRequirements:
the build kernel derives the default minimums from the topology (single
process -> no minimums; multi-worker -> distributed coordination + leasing +
fencing; multi-process swarm -> distributed coordination + fencing + the
multi-process flag) and runs them through the same gate. The gate always
runs -- topology-derived minimums and caller-supplied explicit minimums hit
the same enforcement path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..storage.features import CoordinationScope, StorageFeatures, TransactionScope

if TYPE_CHECKING:
    # Storage is referenced only in the string annotation of
    # enforce_storage_feature_consistency; RuntimeSettings /
    # RuntimeDependencies only in derive_runtime_requirements' signature.
    # All kept lazy to avoid a runtime import cycle.
    from ..runtime.builder import RuntimeSettings
    from ..runtime.dependencies import RuntimeDependencies
    from ..storage.facade import Storage

# Scope ranks: a storage scope must meet-or-exceed the required scope. The two
# enums carry string values whose lexical order is NOT the capability order
# ("distributed" < "none" < "process_local" alphabetically), so rank them
# explicitly: NONE < PROCESS_LOCAL < (DATABASE | DISTRIBUTED).
_COORDINATION_RANK: "dict[CoordinationScope, int]" = {
    CoordinationScope.NONE: 0,
    CoordinationScope.PROCESS_LOCAL: 1,
    CoordinationScope.DISTRIBUTED: 2,
}
_TRANSACTION_RANK: "dict[TransactionScope, int]" = {
    TransactionScope.NONE: 0,
    TransactionScope.PROCESS_LOCAL: 1,
    TransactionScope.DATABASE: 2,
}
# Boolean capabilities a runtime may require. Each defaults to False (no
# requirement); the gate fails only when a requirement is set that storage
# does not offer. ``streaming_artifacts`` / ``optimistic_concurrency`` are
# component-level frozensets on StorageFeatures -- a True requirement is met
# when the frozenset is non-empty (the storage offers the capability for at
# least one component).
_BOOL_CAPABILITIES: "tuple[str, ...]" = (
    "leasing",
    "fencing",
    "idempotency",
    "streaming_artifacts",
    "append_only_events",
)


class RuntimeTopology(str, Enum):
    """The shape of the process graph a Runtime is being assembled for.

    The build kernel uses this ONLY to derive default capability minimums
    when the caller does not pass explicit ``RuntimeRequirements``; it never
    branches behavior on the value beyond that derivation (the capability
    gate stays the single source of truth). A caller that knows its exact
    needs passes ``requirements`` directly and the topology is ignored.

    - SINGLE_PROCESS: one Runtime in one process -- the only Storage it
      touches is its own, so no cross-process capability is required.
    - MULTI_WORKER: multiple workers share JobStore rows -- distributed
      coordination + leasing + fencing are required (a process-local
      coordinator races; an unfenced stale worker's commit is unsafe).
    """

    SINGLE_PROCESS = "single_process"
    MULTI_WORKER = "multi_worker"


def derive_runtime_requirements(
    *,
    settings: "RuntimeSettings",
    dependencies: "RuntimeDependencies",
) -> "RuntimeRequirements":
    """The capability minimums a topology declares by default. Used by the
    build kernel when the caller hands it a topology (on ``settings``) but no
    explicit RuntimeRequirements: the derived minimums are then enforced by the
    same gate an explicit requirements object would hit. A caller that needs
    something different passes ``requirements`` directly and the topology-
    derived defaults are not consulted.

    ``settings.topology`` is the single source for the shape of the process
    graph. ``dependencies`` carries the composition-root objects the topology's
    real-object rules read: a MULTI_WORKER topology demands a real
    Storage (JobStore rows are shared across workers) AND an injected
    RunCommitCoordinator (the build kernel no longer selects one). derive
    refuses a dependencies bundle that is missing what the topology needs --
    the frozen ``(*, settings, dependencies)`` signature exists precisely so
    these composition-root presence rules run here, not in a separate call:

    - SINGLE_PROCESS: no cross-process capability required (process-local
      coordination is acceptable; distributed coordination is not demanded).
    - MULTI_WORKER: DISTRIBUTED coordination + leasing + fencing required (a
      process-local coordinator races on shared JobStore rows; an unfenced
      stale worker's commit is unsafe) AND the dependencies bundle must carry
      a real Storage + RunCommitCoordinator.
    """
    if settings.topology is RuntimeTopology.MULTI_WORKER:
        # Real-object presence the topology's shared-row model requires: a
        # multi-worker deployment cannot run without a Storage (JobStore rows
        # are shared), without a JobStore on that Storage (the rows workers
        # share live there), or without an injected RunCommitCoordinator. These
        # are composition-root rules read off ``dependencies`` -- the capability
        # minimums below are enforced separately by the capability gate.
        if dependencies.storage is None:
            from ..errors import StorageRequirementsNotMetError

            raise StorageRequirementsNotMetError(
                "MULTI_WORKER topology requires a real Storage on dependencies "
                "(JobStore rows are shared across workers); got dependencies."
                "storage=None"
            )
        if dependencies.storage.jobs is None:
            from ..errors import StorageRequirementsNotMetError

            raise StorageRequirementsNotMetError(
                "MULTI_WORKER topology requires a JobStore on the Storage "
                "(workers share JobStore rows); got dependencies.storage."
                "jobs=None"
            )
        if dependencies.run_commit_coordinator is None:
            from ..errors import StorageRequirementsNotMetError

            raise StorageRequirementsNotMetError(
                "MULTI_WORKER topology requires an injected RunCommitCoordinator "
                "on dependencies (the build kernel no longer selects one); got "
                "dependencies.run_commit_coordinator=None"
            )
        return RuntimeRequirements.for_multi_worker_jobs()
    return RuntimeRequirements()


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    """Capability minimums a runtime topology declares it needs from storage."""

    coordination: CoordinationScope = CoordinationScope.NONE
    transactions: TransactionScope = TransactionScope.NONE
    leasing: bool = False
    fencing: bool = False
    idempotency: bool = False
    streaming_artifacts: bool = False
    optimistic_concurrency: bool = False
    append_only_events: bool = False
    # A MULTI_WORKER topology with an enabled ArtifactStore requires its digest
    # coordinator to be DISTRIBUTED (a process-local one races between workers on
    # the put/sweep window). NONE (the default) imposes no requirement -- a
    # single-process topology, or one with no ArtifactStore, is unaffected.
    artifact_coordination: CoordinationScope = CoordinationScope.NONE

    @classmethod
    def for_multi_worker_jobs(cls) -> "RuntimeRequirements":
        """A multi-worker JobRuntime requires DISTRIBUTED coordination, leasing,
        AND fencing -- a process-local coordinator races on shared JobStore rows
        across workers, and each worker's claim must be fenced so a stale
        worker's state commit is rejected. If the Storage also wires an
        ArtifactStore, its digest coordinator must be DISTRIBUTED too --
        artifact_coordination is enforced ONLY when streaming_artifacts is true on
        the storage (checked by the gate, not derivable from the requirements
        object alone, so this classmethod declares it unconditionally and the
        gate skips the check when there is no ArtifactStore)."""
        return cls(
            coordination=CoordinationScope.DISTRIBUTED,
            leasing=True,
            fencing=True,
            artifact_coordination=CoordinationScope.DISTRIBUTED,
        )


def enforce_storage_capability_gate(
    features: StorageFeatures, requirements: "RuntimeRequirements | None"
) -> None:
    """Raise :class:`StorageRequirementsNotMetError` if ``features`` does not
    satisfy ``requirements``. A no-op when ``requirements`` is ``None`` (the
    default, so existing callers impose no gate)."""
    if requirements is None:
        return
    from ..errors import StorageRequirementsNotMetError

    if _COORDINATION_RANK[features.coordination_scope] < _COORDINATION_RANK[
        requirements.coordination
    ]:
        raise StorageRequirementsNotMetError(
            f"storage coordination scope {features.coordination_scope.value!r} is "
            f"below the required {requirements.coordination.value!r}"
        )
    # Artifact-coordination is enforced ONLY when the storage actually has an
    # ArtifactStore wired (streaming_artifacts=True); a storage with no
    # ArtifactStore declares artifact_coordination_scope=NONE by definition and
    # must not be rejected for a topology-derived requirement that assumed
    # artifacts were enabled.
    if features.streaming_artifacts and (
        _COORDINATION_RANK[features.artifact_coordination_scope]
        < _COORDINATION_RANK[requirements.artifact_coordination]
    ):
        raise StorageRequirementsNotMetError(
            "storage artifact coordination scope "
            f"{features.artifact_coordination_scope.value!r} is below the "
            f"required {requirements.artifact_coordination.value!r}"
        )
    if _TRANSACTION_RANK[features.transaction_scope] < _TRANSACTION_RANK[
        requirements.transactions
    ]:
        raise StorageRequirementsNotMetError(
            f"storage transaction scope {features.transaction_scope.value!r} is "
            f"below the required {requirements.transactions.value!r}"
        )
    for flag in _BOOL_CAPABILITIES:
        if getattr(requirements, flag) and not getattr(features, flag):
            raise StorageRequirementsNotMetError(
                f"storage does not provide the required capability {flag!r}"
            )
    # optimistic_concurrency is component-level on features (frozenset) but a
    # bool on requirements: a True requirement is met when the storage offers
    # optimistic concurrency for at least one component.
    if requirements.optimistic_concurrency and not features.optimistic_concurrency:
        raise StorageRequirementsNotMetError(
            "storage does not provide the required capability 'optimistic_concurrency'"
        )


def enforce_storage_feature_consistency(storage: "Storage") -> None:
    """Verify a Storage's declared :class:`StorageFeatures` match its WIRED
    objects. This catches a backend that DECLARES a capability on its features
    but did not actually wire the object backing it -- a silent degradation the
    requirements-vs-features gate cannot see (it only reads the feature flags).

    Checks:
    * transaction_scope != NONE  -> _transaction_manager must be a real manager
      (not NoCrossStoreTransactions), and a DATABASE scope must not use a
      NoCrossStoreTransactions manager (which cannot group stores).
    * coordination_scope != NONE  -> storage.coordination must be a real coordinator.
    * streaming_artifacts=True  -> storage.artifacts must be a real ArtifactStore
      (and the reverse: an ArtifactStore wired but streaming_artifacts=False is a
      flag that disagrees with the wired store).
    * leasing=True          -> coordination present.
    * transactional_components -> each declared component must have a real wired
      store on the Storage (a declared-but-unwired component is a false claim).
    * JOBS in transactional_components -> storage.jobs must be non-None.

    Run this at build_runtime alongside the requirements gate so a
    misconfigured Storage fails fast instead of producing an AttributeError at
    the first use."""
    from ..errors import StorageRequirementsNotMetError
    from ..storage.features import StorageComponent
    from ..storage.transaction import NoCrossStoreTransactions

    f = storage.features
    if f.transaction_scope is not TransactionScope.NONE:
        if storage._transaction_manager is None:
            raise StorageRequirementsNotMetError(
                f"Storage declares transaction_scope={f.transaction_scope.value!r} "
                "but its transaction manager is None"
            )
        if (
            f.transaction_scope is TransactionScope.DATABASE
            and isinstance(storage._transaction_manager, NoCrossStoreTransactions)
        ):
            raise StorageRequirementsNotMetError(
                "Storage declares transaction_scope=DATABASE but its transaction "
                "manager is NoCrossStoreTransactions (cannot group stores)"
            )
    if (
        f.coordination_scope is not CoordinationScope.NONE
        and storage.coordination is None
    ):
        raise StorageRequirementsNotMetError(
            f"Storage declares coordination_scope={f.coordination_scope.value!r} "
            "but its LeaseCoordinator is None"
        )
    if f.streaming_artifacts and storage.artifacts is None:
        raise StorageRequirementsNotMetError(
            "Storage declares streaming_artifacts=True but its ArtifactStore "
            "(storage.artifacts) is None"
        )
    if storage.artifacts is not None and not f.streaming_artifacts:
        # The reverse direction: an ArtifactStore is wired (it can stream) but
        # the flag says it cannot. The flag must agree with the wired store.
        raise StorageRequirementsNotMetError(
            "Storage wires an ArtifactStore (storage.artifacts) but declares "
            "streaming_artifacts=False -- the flag must agree with the wired store"
        )
    wired_coordinator = getattr(storage.artifacts, "_coordinator", None)
    if wired_coordinator is not None:
        # The declared artifact_coordination_scope must match the ACTUAL wired
        # coordinator's own scope -- a mismatch here is a Storage that claims one
        # capability but wired a coordinator providing a different one (the
        # topology gate above only sees the claim, not the real object).
        wired_scope = wired_coordinator.scope
        if wired_scope != f.artifact_coordination_scope:
            raise StorageRequirementsNotMetError(
                f"Storage declares artifact_coordination_scope="
                f"{f.artifact_coordination_scope.value!r} but its wired "
                f"ArtifactDigestCoordinator scope is {wired_scope.value!r}"
            )
    if f.leasing and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            "Storage declares leasing=True but its LeaseCoordinator is None "
            "(acquire/renew/release need a coordinator)"
        )
    if f.fencing and storage.coordination is None:
        # Fencing tokens are minted by the coordinator; a fencing=True
        # declaration with no coordinator is a capability with no backing.
        raise StorageRequirementsNotMetError(
            "Storage declares fencing=True but its LeaseCoordinator is None "
            "(fencing tokens are minted by the coordinator)"
        )
    # Each declared transactional component must be backed by a real wired
    # store; a declared component with no store is a capability the Storage
    # cannot actually deliver inside a transaction.
    wired = _wired_components(storage)
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


def _wired_components(storage: "Storage") -> "frozenset[StorageComponent]":
    """The set of StorageComponents that have a non-None wired store on this
    Storage. ``assets`` / ``jobs`` are optional on the Storage facade and read
    through their accessors; the core stores are always present."""
    from ..storage.features import StorageComponent

    wired: "set[StorageComponent]" = {
        StorageComponent.RUNS,
        StorageComponent.SESSIONS,
        StorageComponent.EVENTS,
        StorageComponent.APPROVALS,
        StorageComponent.CHECKPOINTS,
        StorageComponent.ARTIFACT_RECORDS,
    }
    if storage.assets is not None:
        wired.add(StorageComponent.ASSETS)
    if storage.jobs is not None:
        wired.add(StorageComponent.JOBS)
    return frozenset(wired)


__all__: "list[str]" = [
    "RuntimeTopology",
    "RuntimeRequirements",
    "derive_runtime_requirements",
    "enforce_storage_capability_gate",
    "enforce_storage_feature_consistency",
]
