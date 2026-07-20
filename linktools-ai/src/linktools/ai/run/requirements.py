#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeRequirements: what a runtime topology NEEDS from storage, declared as
capability minimums. The RuntimeBuilder capability gate compares these against
:class:`~linktools.ai.storage.features.StorageFeatures` (what storage
DECLARES) and fails fast at build time on a shortfall -- never silently
degrading to a weaker scope.

This is the plan's Phase 9 op 6 capability gate ("纯 capability 门禁"): it
branches on capability scope/flag values, never on ``isinstance`` against a
concrete backend. A caller whose topology needs, e.g., distributed
coordination passes
``Runtime.build(requirements=RuntimeRequirements(coordination=CoordinationScope.DISTRIBUTED))``
and the builder refuses a process-local Storage rather than letting it race.

Every field defaults to the most permissive value (no requirement), so a bare
``RuntimeRequirements()`` -- or passing ``None`` -- imposes no gate. Callers
opt INTO enforcement by raising the minimums their topology actually needs;
subsystems (a multi-worker JobRuntime, a multi-process Swarm) declare their
own requirements in their own wiring."""

from __future__ import annotations

from dataclasses import dataclass

from ..storage.features import CoordinationScope, StorageFeatures, TransactionScope

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
    # A multi-process Swarm topology: separate processes cannot share a
    # process-local coordinator, and Swarm state commits must be fenced. When
    # True the gate requires DISTRIBUTED coordination AND fencing -- on top of
    # whatever the scalar coordination scope already enforces.
    multi_process_swarm: bool = False

    @classmethod
    def for_multi_worker_jobs(cls) -> "RuntimeRequirements":
        """Plan §4.11: a multi-worker JobRuntime requires DISTRIBUTED
        coordination, leasing, AND fencing -- a process-local coordinator
        races on shared JobStore rows across workers, and each worker's claim
        must be fenced so a stale worker's state commit is rejected."""
        return cls(
            coordination=CoordinationScope.DISTRIBUTED,
            leasing=True,
            fencing=True,
        )

    @classmethod
    def for_multi_process_swarm(cls) -> "RuntimeRequirements":
        """Plan §4.11: a multi-process Swarm requires DISTRIBUTED coordination
        AND fencing. Sets the ``multi_process_swarm`` flag so the gate enforces
        the combination explicitly (not just the scalar coordination scope)."""
        return cls(
            coordination=CoordinationScope.DISTRIBUTED,
            fencing=True,
            multi_process_swarm=True,
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
    # A multi-process Swarm topology requires DISTRIBUTED coordination AND
    # fencing. The scalar coordination check above already catches a plain
    # scope shortfall, but the multi_process_swarm flag makes the topology
    # intent explicit and lets the gate reject a backend that meets the scope
    # yet lacks fencing.
    if requirements.multi_process_swarm:
        if (
            _COORDINATION_RANK[features.coordination]
            < _COORDINATION_RANK[CoordinationScope.DISTRIBUTED]
        ):
            raise StorageRequirementsNotMetError(
                f"multi-process swarm requires DISTRIBUTED coordination; "
                f"storage provides {features.coordination.value!r}"
            )
        if not features.fencing:
            raise StorageRequirementsNotMetError(
                "multi-process swarm requires fencing"
            )


__all__: "list[str]" = ["RuntimeRequirements", "enforce_storage_capability_gate"]
