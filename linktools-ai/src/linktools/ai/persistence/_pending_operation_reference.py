#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending calls borrow immutable admission fields from their local effect owner."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from ..core import validate_resource_id
from ..storage import StoredPayload
from ..spec import CapabilityRef

from ..errors import AIError, ErrorCode
from .contracts import PendingOperationSource, ToolOperationRecord
from .codec import _CURRENT_CODEC, _validate_dataclass_wrapper, decode_domain, encode_envelope, decode_envelope, execution_payload_slots, pending_operation_reference
from .state import StateStore, StoredRecord, record_key
from ._tool_operation_content import ToolOperationContent


def decode_pending_operation(record: StoredRecord, *, namespace: str, tenant_id: str) -> PendingOperationSource:
    """Decode only immutable admission fields, never the terminal result/error."""
    try:
        envelope = decode_envelope(record.data)
        payload = envelope.value["payload"]
        if (envelope.value["type"] != "tool_operation" or payload["$dataclass"] != "tool_operation"
                or type(payload["schema"]) is not int or payload["schema"] != 1):
            raise ValueError("invalid pending operation source")
        fields = payload["fields"]
        identity = fields["tool_operation_id"]
        for key in ("tool_operation_id", "execution_id", "agent_run_id", "tool_call_id"):
            validate_resource_id(fields[key])
        if not isinstance(fields["tool_name"], str) or not fields["tool_name"]:
            raise ValueError("invalid tool name")
        arguments = decode_domain(fields["arguments_payload"], StoredPayload)
        if arguments.digest != fields["arguments_digest"]:
            raise ValueError("tool arguments digest mismatch")
        capability = fields["capability_ref"]
        value = PendingOperationSource(identity, fields["execution_id"], fields["tool_call_id"],
            fields["tool_name"], arguments, None if capability is None else decode_domain(capability, CapabilityRef))
        if (record.kind != "tool_operation" or record.subject_id != value.tool_call_id
                or record.key != record_key(namespace, tenant_id, "checkpoint", "tool_operation", (identity,))):
            raise ValueError("pending operation address mismatch")
        return value
    except (ValueError, TypeError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _arguments_only(record: StoredRecord) -> StoredRecord:
    envelope = decode_envelope(record.data)
    payload = envelope.value.get("payload")
    _, fields = _validate_dataclass_wrapper(payload, ToolOperationRecord, _CURRENT_CODEC, persisted=True)
    return replace(record, data=encode_envelope({**envelope.value, "payload": {**payload,
        "fields": {**fields, "result_payload": None, "error_payload": None}}}))


async def read_pending_operations(store: StateStore, identities: Sequence[str], *,
                                  namespace: str, tenant_id: str) -> tuple[PendingOperationSource, ...]:
    keys = tuple(record_key(namespace, tenant_id, "checkpoint", "tool_operation", (identity,))
                 for identity in dict.fromkeys(identities))
    rows = await store.read(lambda transaction: transaction.get_records(keys))
    content = ToolOperationContent(store, namespace=namespace, tenant_id=tenant_id)
    expanded = await content.expand_many(tuple(_arguments_only(rows[key]) for key in keys if key in rows))
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
