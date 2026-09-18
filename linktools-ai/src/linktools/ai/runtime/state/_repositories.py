#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository composition and stable implementation exports."""

from ._conversation_repositories import ConversationHistoryRepositoryImpl, SessionRepositoryImpl
from ._execution_repositories import EventRepositoryImpl, ExecutionRepositoryImpl, IdempotencyRepositoryImpl
from ._plan import RuntimeDomain
from ._recovery_repositories import (
    ApprovalRepositoryImpl, ExternalCallRepositoryImpl, RecoveryCheckpointRepositoryImpl, ToolRepositoryImpl,
)
from ._repository_common import (
    OperationLedgerRepository, RepositoryBase, append_operation, decode_operation, decode_record_cursor,
    projected_record, record_cursor, replace_checked, require_repository_tenant,
)
from ._resource_repositories import ArtifactRepositoryImpl, EvaluationRepositoryImpl, MemoryRepositoryImpl
from ._store import StateStore


def build_repository_bundle(
    store: StateStore, *, namespace: str, tenant_id: str, domain: RuntimeDomain
) -> dict[str, RepositoryBase]:
    """Build the one semantic repository implementation for a domain."""
    values: dict[str, RepositoryBase] = {
        "operations": OperationLedgerRepository(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
    }
    if domain is RuntimeDomain.CONVERSATION:
        values.update(
            sessions=SessionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            histories=ConversationHistoryRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        )
    elif domain is RuntimeDomain.EXECUTION:
        values.update(
            executions=ExecutionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            events=EventRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is RuntimeDomain.MEMORY:
        values["records"] = MemoryRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is RuntimeDomain.ARTIFACT:
        values["records"] = ArtifactRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is RuntimeDomain.EVALUATION:
        values.update(
            records=EvaluationRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is RuntimeDomain.RECOVERY:
        values.update(
            approvals=ApprovalRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            external_calls=ExternalCallRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            checkpoints=RecoveryCheckpointRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            tools=ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        )
    return values


__all__ = [
    "ApprovalRepositoryImpl", "ArtifactRepositoryImpl", "EvaluationRepositoryImpl", "EventRepositoryImpl",
    "ExecutionRepositoryImpl", "ExternalCallRepositoryImpl", "IdempotencyRepositoryImpl", "MemoryRepositoryImpl",
    "OperationLedgerRepository", "SessionRepositoryImpl", "ToolRepositoryImpl", "RepositoryBase",
    "append_operation", "build_repository_bundle", "decode_operation", "decode_record_cursor",
    "projected_record", "record_cursor", "replace_checked", "require_repository_tenant",
]
