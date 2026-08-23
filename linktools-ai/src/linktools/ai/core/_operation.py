"""Generic durable operation ledger values."""

from dataclasses import dataclass
from datetime import datetime

from ._value import OperationKind, OperationStatus, ResourceKind


@dataclass(frozen=True, slots=True)
class OperationLedgerRecord:
    operation_id: str
    tenant_id: str
    resource_kind: ResourceKind
    resource_id: str
    execution_id: "str | None"
    operation_kind: OperationKind
    status: OperationStatus
    request_digest: str
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"
    compactable: bool
    sequence: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class OperationLedgerInput:
    operation_id: str
    tenant_id: str
    resource_kind: ResourceKind
    resource_id: str
    execution_id: "str | None"
    operation_kind: OperationKind
    status: OperationStatus
    request_digest: str
    result_ref: "str | None"
    result_digest: "str | None"
    error_code: "str | None"
    compactable: bool
    created_at: datetime
    updated_at: datetime


def operation_replay_matches(
    record: OperationLedgerRecord,
    candidate: OperationLedgerInput,
) -> bool:
    return (
        record.operation_id == candidate.operation_id
        and record.tenant_id == candidate.tenant_id
        and record.resource_kind is candidate.resource_kind
        and record.resource_id == candidate.resource_id
        and record.execution_id == candidate.execution_id
        and record.operation_kind is candidate.operation_kind
        and record.status is candidate.status
        and record.request_digest == candidate.request_digest
        and record.result_ref == candidate.result_ref
        and record.result_digest == candidate.result_digest
        and record.error_code == candidate.error_code
        and record.compactable == candidate.compactable
    )


def operation_cas_immutable_matches(
    current: OperationLedgerRecord,
    candidate: OperationLedgerRecord,
) -> bool:
    return (
        current.operation_id == candidate.operation_id
        and current.tenant_id == candidate.tenant_id
        and current.resource_kind is candidate.resource_kind
        and current.resource_id == candidate.resource_id
        and current.execution_id == candidate.execution_id
        and current.operation_kind is candidate.operation_kind
        and current.request_digest == candidate.request_digest
        and current.compactable == candidate.compactable
        and current.sequence == candidate.sequence
        and current.created_at == candidate.created_at
    )


__all__ = [
    "OperationLedgerInput",
    "OperationLedgerRecord",
    "operation_cas_immutable_matches",
    "operation_replay_matches",
]
