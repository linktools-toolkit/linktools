#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending calls borrow immutable admission fields from their local effect owner."""

from collections.abc import Mapping, Sequence

from ..errors import AIError, ErrorCode
from .contracts import ToolOperationRecord
from .codec import _decode_enveloped_domain, decode_envelope, execution_payload_slots, pending_operation_reference
from .state import StateStore, StoredRecord, record_key
from ._tool_operation_content import ToolOperationContent


def decode_pending_operation(record: StoredRecord, *, namespace: str, tenant_id: str) -> ToolOperationRecord:
    def restore(payload: Mapping) -> Mapping:
        return {**payload, "fields": {**payload["fields"], "owner": record.lease_owner, "fence": record.lease_fence}}
    value = _decode_enveloped_domain(record.data, ToolOperationRecord, payload_transform=restore)
    if (record.kind != "tool_operation" or record.subject_id != value.tool_call_id
            or record.key != record_key(namespace, tenant_id, "checkpoint", "tool_operation", (value.tool_operation_id,))):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


async def read_pending_operations(store: StateStore, identities: Sequence[str], *,
                                  namespace: str, tenant_id: str) -> tuple[ToolOperationRecord, ...]:
    keys = tuple(record_key(namespace, tenant_id, "checkpoint", "tool_operation", (identity,))
                 for identity in dict.fromkeys(identities))
    rows = await store.read(lambda transaction: transaction.get_records(keys))
    content = ToolOperationContent(store, namespace=namespace, tenant_id=tenant_id)
    expanded = await content.expand_many(tuple(rows[key] for key in keys if key in rows))
    return tuple(decode_pending_operation(row, namespace=namespace, tenant_id=tenant_id) for row in expanded)


def pending_operation_values(record: StoredRecord, owners: Mapping[bytes, StoredRecord], *,
                             namespace: str, tenant_id: str) -> dict:
    """Resolve a portable owner's references from its complete physical inventory."""
    if record.kind != "execution":
        return {}
    fields = decode_envelope(record.data).value["payload"]["fields"]
    values = {}
    for _, payload in execution_payload_slots(decode_envelope(record.data).value["payload"]):
        identity = pending_operation_reference(payload)
        if identity is None:
            continue
        key = record_key(namespace, tenant_id, "checkpoint", "tool_operation", (identity,))
        source = owners.get(key)
        if source is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        operation = decode_pending_operation(source, namespace=namespace, tenant_id=tenant_id)
        if operation.execution_id != fields["execution_id"] or operation.arguments_payload is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values[("pending-operation", identity)] = operation
    return values


__all__ = ["decode_pending_operation", "read_pending_operations", "pending_operation_values"]
