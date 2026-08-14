#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


__all__ = ["OperationLedgerInput", "OperationLedgerRecord"]
