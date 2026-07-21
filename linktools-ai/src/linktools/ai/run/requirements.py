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
``Runtime.build(requirements=RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED))``
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
    from .._runtime.build import RuntimeSettings
    from .._runtime.dependencies import RuntimeDependencies
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
# does not offer.
_BOOL_CAPABILITIES: "tuple[str, ...]" = (
    "leasing",
    "fencing",
    "idempotency",
    "streaming_blobs",
    "optimistic_concurrency",
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
    real-object rules read (plan §6.6): a MULTI_WORKER topology demands a real
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
      a real Storage + RunCommitCoordinator (plan: "JobStore 必须存在;
      coordinator 必须存在").
    """
    if settings.topology is RuntimeTopology.MULTI_WORKER:
        # Real-object presence the topology's shared-row model requires: a
        # multi-worker deployment cannot run without a Storage (JobStore rows
        # are shared) or without an injected RunCommitCoordinator. These are
        # composition-root rules read off ``dependencies`` -- the capability
        # minimums below are enforced separately by the capability gate.
        if dependencies.storage is None:
            from ..errors import StorageRequirementsNotMetError

            raise StorageRequirementsNotMetError(
                "MULTI_WORKER topology requires a real Storage on dependencies "
                "(JobStore rows are shared across workers); got dependencies."
                "storage=None"
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
    streaming_blobs: bool = False
    optimistic_concurrency: bool = False
    append_only_events: bool = False

    @classmethod
    def for_multi_worker_jobs(cls) -> "RuntimeRequirements":
        """A multi-worker JobRuntime requires DISTRIBUTED coordination, leasing,
        AND fencing -- a process-local coordinator races on shared JobStore rows
        across workers, and each worker's claim must be fenced so a stale
        worker's state commit is rejected."""
        return cls(
            coordination=CoordinationScope.DISTRIBUTED,
            leasing=True,
            fencing=True,
        )


def enforce_storage_capability_gate(
    features: StorageFeatures, requirements: "RuntimeRequirements | None"
) -> None:
    """Raise :class:`StorageRequirementsNotMetError` if ``features`` does not
    satisfy ``requirements``. A no-op when ``requirements`` is ``None`` (the
    default, backward-compatible path -- existing callers impose no gate)."""
    if requirements is None:
        return
    from ..errors import StorageRequirementsNotMetError

    if _COORDINATION_RANK[features.coordination] < _COORDINATION_RANK[
        requirements.coordination
    ]:
        raise StorageRequirementsNotMetError(
            f"storage coordination scope {features.coordination.value!r} is "
            f"below the required {requirements.coordination.value!r}"
        )
    if _TRANSACTION_RANK[features.transactions] < _TRANSACTION_RANK[
        requirements.transactions
    ]:
        raise StorageRequirementsNotMetError(
            f"storage transaction scope {features.transactions.value!r} is "
            f"below the required {requirements.transactions.value!r}"
        )
    for flag in _BOOL_CAPABILITIES:
        if getattr(requirements, flag) and not getattr(features, flag):
            raise StorageRequirementsNotMetError(
                f"storage does not provide the required capability {flag!r}"
            )


def enforce_storage_feature_consistency(storage: "Storage") -> None:
    """Verify a Storage's declared :class:`StorageFeatures` match its WIRED
    objects (plan §6.6: '只校验 StorageFeatures 不够，还必须检查真实对象'). This
    catches a backend that DECLARES a capability on its features but did not
    actually wire the object backing it -- a silent degradation the
    requirements-vs-features gate cannot see (it only reads the feature flags).

    Checks:
    * transactions != NONE  -> storage.transactions must be a real manager.
    * coordination != NONE  -> storage.coordination must be a real coordinator.
    * streaming_blobs=True  -> storage.artifacts must be a real ArtifactStore.
    * leasing=True          -> coordination present (acquire/renew/release live
      on the coordinator; a None coordinator cannot lease).

    Run this at Runtime.build alongside the requirements gate so a
    misconfigured Storage fails fast instead of producing an AttributeError at
    the first use."""
    from ..errors import StorageRequirementsNotMetError

    f = storage.features
    if f.transactions is not TransactionScope.NONE and storage.transactions is None:
        raise StorageRequirementsNotMetError(
            f"Storage declares transactions={f.transactions.value!r} but its "
            "transactions manager is None"
        )
    if f.coordination is not CoordinationScope.NONE and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            f"Storage declares coordination={f.coordination.value!r} but its "
            "LeaseCoordinator is None"
        )
    if f.streaming_blobs and storage.artifacts is None:
        raise StorageRequirementsNotMetError(
            "Storage declares streaming_blobs=True but its ArtifactStore "
            "(storage.artifacts) is None"
        )
    if f.leasing and storage.coordination is None:
        raise StorageRequirementsNotMetError(
            "Storage declares leasing=True but its LeaseCoordinator is None "
            "(acquire/renew/release need a coordinator)"
        )


__all__: "list[str]" = [
    "RuntimeTopology",
    "RuntimeRequirements",
    "derive_runtime_requirements",
    "enforce_storage_capability_gate",
    "enforce_storage_feature_consistency",
]
