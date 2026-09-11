#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery-domain repository implementations with bounded batch I/O."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from ...core import ApprovalStatus, ExternalCallStatus
from ...errors import AIError, ErrorCode
from ._contracts import ApprovalRecord, ExternalCallRecord
from ._plan import RuntimeDomain
from ._repositories import (
    ApprovalRepositoryImpl as _ApprovalRepositoryImpl,
    ExternalCallRepositoryImpl as _ExternalCallRepositoryImpl,
    _RepositoryBase,
    build_repository_bundle as _build_repository_bundle,
)
from ._store import RecordReplacement, StateStore, StateTransaction


class RecoveryApprovalRepositoryImpl(_ApprovalRepositoryImpl):
    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        approval_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        decided_at: datetime,
    ) -> tuple[ApprovalRecord, ...]:
        _validate_cancel_request(approval_ids, tenant_id, self._tenant_id, decided_at)
        ordered = tuple(dict.fromkeys(approval_ids))
        if not ordered:
            return ()
        keys = tuple(self._key("approval", approval_id) for approval_id in ordered)
        stored_records = await transaction.get_records(keys)
        values: list[ApprovalRecord] = []
        replacements: list[RecordReplacement] = []
        for approval_id, key in zip(ordered, keys, strict=True):
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(stored, ApprovalRecord)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is not ApprovalStatus.PENDING:
                values.append(current)
                continue
            value = replace(
                current,
                status=ApprovalStatus.CANCELLED,
                decided_at=decided_at,
            )
            replacements.append(
                RecordReplacement(
                    replace(
                        self._stored(
                            "approval",
                            value.approval_id,
                            value,
                            state=value.status.value,
                        ),
                        storage_version=stored.storage_version + 1,
                    ),
                    stored.storage_version,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
        return tuple(values)


class RecoveryExternalCallRepositoryImpl(_ExternalCallRepositoryImpl):
    async def cancel_pending_in_transaction(
        self,
        transaction: StateTransaction,
        call_ids: Sequence[str],
        *,
        execution_id: str,
        tenant_id: str,
        cancelled_at: datetime,
    ) -> tuple[ExternalCallRecord, ...]:
        _validate_cancel_request(call_ids, tenant_id, self._tenant_id, cancelled_at)
        ordered = tuple(dict.fromkeys(call_ids))
        if not ordered:
            return ()
        keys = tuple(self._key("external_call", call_id) for call_id in ordered)
        stored_records = await transaction.get_records(keys)
        values: list[ExternalCallRecord] = []
        replacements: list[RecordReplacement] = []
        for call_id, key in zip(ordered, keys, strict=True):
            stored = stored_records.get(key)
            if stored is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            current = await self._decode(stored, ExternalCallRecord)
            if current.execution_id != execution_id:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if current.status is not ExternalCallStatus.PENDING:
                values.append(current)
                continue
            value = replace(
                current,
                status=ExternalCallStatus.CANCELLED,
                supplied_at=cancelled_at,
            )
            replacements.append(
                RecordReplacement(
                    replace(
                        self._stored(
                            "external_call",
                            value.call_id,
                            value,
                            state=value.status.value,
                        ),
                        storage_version=stored.storage_version + 1,
                    ),
                    stored.storage_version,
                )
            )
            values.append(value)
        if replacements:
            await transaction.replace_records(tuple(replacements))
        return tuple(values)


def build_recovery_repository_bundle(
    store: StateStore,
    *,
    namespace: str,
    tenant_id: str,
) -> dict[str, _RepositoryBase]:
    values = _build_repository_bundle(
        store,
        namespace=namespace,
        tenant_id=tenant_id,
        domain=RuntimeDomain.RECOVERY,
    )
    values["approvals"] = RecoveryApprovalRepositoryImpl(
        store,
        namespace=namespace,
        tenant_id=tenant_id,
    )
    values["external_calls"] = RecoveryExternalCallRepositoryImpl(
        store,
        namespace=namespace,
        tenant_id=tenant_id,
    )
    return values


def _validate_cancel_request(
    identities: Sequence[str],
    tenant_id: str,
    expected_tenant_id: str,
    timestamp: datetime,
) -> None:
    if tenant_id != expected_tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("identities must be a sequence")
    if any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("identities must contain non-empty strings")
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")


__all__: list[str] = []
