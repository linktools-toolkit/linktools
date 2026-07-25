#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable runtime-persistence extension Protocols -- the public surface a
downstream or external adapter implements to plug into the Runtime.

These Protocols depend only on the standard library and ``linktools-ai``
domain models. A backend (Filesystem, SQLAlchemy, or an external one) implements
them; the RuntimeBuilder capability-gates on StorageFeatures + Protocol
availability, never on ``isinstance`` against a concrete class. External
adapters hide multipart uploads, connection pools, retries and vendor errors
behind these Protocols; they convert all exceptions to core error types via
``raise ... from exc``.

``StorageUnitOfWork`` is the cross-store atomic handle: it types each store
field as the concrete store Protocol (RunStore / SessionStore / EventStore /
...) so a downstream consumer gets real attribute-completion checking. The
optional stores (artifact_records, jobs) are explicit ``X | None`` rather
than ``Any`` so a backend that does not provide one declares the gap
honestly, not via an absent attribute.

The coordination Protocols (``LeaseCoordinator`` + ``LeaseToken`` +
``KeyedCoordinator``) live at ``linktools.ai.storage.coordination.protocols``
-- they are storage-kernel machinery with no domain coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncContextManager, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Protocol-level imports kept TYPE_CHECKING-only so the
    # ``runtime.persistence.protocols`` module has no runtime dependency on
    # the concrete store modules (which would re-introduce an import cycle
    # runtime.persistence.protocols -> artifact.persistence -> artifact
    # (.__init__ imports .store) -> runtime.persistence.protocols). External
    # adapters can ``from linktools.ai.runtime.persistence.protocols import
    # ...`` in any import order without hitting a partially-initialized
    # module.
    from ...agent.approval import ApprovalStore
    from ...artifact.persistence.protocols import ArtifactRecordStore
    from ...events.store import EventStore
    from ...jobs.store import JobStore
    from ...run.checkpoint import CheckpointStore
    from ...run.store import RunStore
    from ...session.store import SessionStore
    from ...storage.object.store import ObjectStore
    from ...tool.idempotency import IdempotencyStore


@runtime_checkable
class StorageTransactionManager(Protocol):
    """Cross-store atomic scope. ``transaction()`` commits once on clean exit
    and rolls back once on exception; callers never call backend commit/rollback
    directly. An unsupported scope raises StorageTransactionNotSupportedError at
    the call."""

    def transaction(self) -> AsyncContextManager["StorageUnitOfWork"]: ...


@runtime_checkable
class StorageUnitOfWork(Protocol):
    """The stores sharing one transaction.

    Every field carries its concrete Protocol type so a downstream consumer
    (run commit coordinator, runtime) gets real attribute-completion checking
    against the union of stores a backend actually exposes. Optional stores
    that a backend does not provide in this scope are declared as an explicit
    ``X | None`` -- never an untyped ``Any`` and never an absent attribute
    (an optional store + None is the honest declaration; an absent attribute
    hides the capability gap from type-checking).
    """

    # assets is a session-bound ObjectStore in every transactional UoW: the
    # backend reuses the UoW's session so asset mutations commit or roll back
    # with every other store. A backend that cannot bind assets to the
    # transaction does not offer a cross-store UoW at all (it declares
    # TransactionScope.NONE and its transaction() raises).
    assets: "ObjectStore"
    artifact_records: "ArtifactRecordStore"
    sessions: "SessionStore"
    runs: "RunStore"
    events: "EventStore"
    checkpoints: "CheckpointStore"
    approvals: "ApprovalStore"
    idempotency: "IdempotencyStore"
    jobs: "JobStore | None"


__all__: "list[str]" = [
    "StorageTransactionManager",
    "StorageUnitOfWork",
]
