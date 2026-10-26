#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository composition and stable implementation exports."""

from linktools.ai.persistence._budget import BudgetRepositoryImpl
from linktools.ai.persistence._conversation_repositories import (
    ConversationHistoryRepositoryImpl,
    SessionRepositoryImpl,
)
from linktools.ai.persistence._execution_repositories import (
    EventRepositoryImpl,
    ExecutionRepositoryImpl,
    IdempotencyRepositoryImpl,
)
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._recovery_repositories import (
    ApprovalRepositoryImpl, ExternalCallRepositoryImpl, RecoveryCheckpointRepositoryImpl, ToolRepositoryImpl,
)
from linktools.ai.persistence._repository_common import (
    OperationLedgerRepositoryImpl, RepositoryBase, append_operation, decode_operation, decode_record_cursor,
    projected_record, record_cursor, replace_checked, require_repository_tenant,
)
from linktools.ai.persistence._evaluation_repository import EvaluationRepositoryImpl
from linktools.ai.persistence._resource_repositories import ArtifactRepositoryImpl, MemoryRepositoryImpl
from linktools.ai.persistence.state import StateStore


def build_repository_bundle(
    store: StateStore, *, namespace: str, tenant_id: str, domain: PersistenceDomain
) -> dict[str, RepositoryBase]:
    """Build the one semantic repository implementation for a domain."""
    values: dict[str, RepositoryBase] = {
        "operations": OperationLedgerRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain)
    }
    if domain is PersistenceDomain.CONVERSATION:
        values.update(
            sessions=SessionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            histories=ConversationHistoryRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        )
    elif domain is PersistenceDomain.EXECUTION:
        values.update(
            budgets=BudgetRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            executions=ExecutionRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            events=EventRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is PersistenceDomain.MEMORY:
        values["records"] = MemoryRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is PersistenceDomain.ARTIFACT:
        values["records"] = ArtifactRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id)
    elif domain is PersistenceDomain.EVALUATION:
        values.update(
            records=EvaluationRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            idempotency=IdempotencyRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id, domain=domain),
        )
    elif domain is PersistenceDomain.RECOVERY:
        values.update(
            approvals=ApprovalRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            external_calls=ExternalCallRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            checkpoints=RecoveryCheckpointRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
            tools=ToolRepositoryImpl(store, namespace=namespace, tenant_id=tenant_id),
        )
    return values


from ._repository_common import operation_identity_matches

__all__ = ['ApprovalRepositoryImpl', 'ArtifactRepositoryImpl', 'BudgetRepositoryImpl', 'ConversationHistoryRepositoryImpl', 'EvaluationRepositoryImpl', 'EventRepositoryImpl', 'ExecutionRepositoryImpl', 'ExternalCallRepositoryImpl', 'IdempotencyRepositoryImpl', 'MemoryRepositoryImpl', 'OperationLedgerRepositoryImpl', 'RecoveryCheckpointRepositoryImpl', 'RepositoryBase', 'SessionRepositoryImpl', 'ToolRepositoryImpl', 'append_operation', 'build_repository_bundle', 'decode_operation', 'decode_record_cursor', 'operation_identity_matches', 'projected_record', 'record_cursor', 'replace_checked', 'require_repository_tenant']
