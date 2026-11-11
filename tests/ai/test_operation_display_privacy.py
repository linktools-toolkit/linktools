#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Operation display values do not reveal durable idempotency keys."""

from dataclasses import replace

from linktools.ai.core import ImmutableJsonMapping
from linktools.ai.persistence.codec import decode_operation, decode_record, encode_operation, encode_record
from linktools.ai.persistence.state import OperationScanCursor, StoredOperation, StoredRecord, operation_key


class _ReadableMapping(ImmutableJsonMapping):
    def __repr__(self) -> str:
        return repr(dict(self))


def test_stored_operation_hides_identity_and_payload_without_changing_wire() -> None:
    raw_key = "sensitive-key-MUST-NOT-APPEAR-IN-REPR"
    payload_key = "sensitive-payload-id-MUST-NOT-APPEAR-IN-REPR"
    data = _ReadableMapping({"operation_id": payload_key})
    assert payload_key in repr(data)
    key = operation_key(raw_key)
    operation = StoredOperation(key, b"s" * 32, 1, "PENDING", False, data)

    assert raw_key not in repr(operation)
    assert payload_key not in repr(operation)
    assert operation.key == key
    assert operation.data["operation_id"] == payload_key
    wire = encode_operation(operation)
    assert wire["key"] == key.hex()
    assert wire["data"] == {"operation_id": payload_key}
    assert decode_operation(wire) == operation
    assert replace(operation, key=operation_key("different")) != operation
    assert replace(operation, data={"operation_id": "different"}) != operation


def test_operation_scan_cursor_hides_identity_without_changing_comparison() -> None:
    raw_key = "sensitive-cursor-key-MUST-NOT-APPEAR-IN-REPR"
    key = operation_key(raw_key)
    cursor = OperationScanCursor(key)

    assert raw_key not in repr(cursor)
    assert cursor.key == key
    assert cursor == OperationScanCursor(key)
    assert cursor != OperationScanCursor(operation_key("different"))


def test_stored_record_hides_operation_payload_without_changing_wire() -> None:
    raw_key = "sensitive-admission-key-MUST-NOT-APPEAR-IN-REPR"
    data = _ReadableMapping({"operation_id": raw_key})
    assert raw_key in repr(data)
    record = StoredRecord(
        b"r" * 32, None, None, "task_admission", "graph", None, 0, None, 0, None, data,
    )

    assert raw_key not in repr(record)
    assert record.data["operation_id"] == raw_key
    wire = encode_record(record)
    assert wire["data"] == {"operation_id": raw_key}
    assert decode_record(wire) == record
    assert replace(record, data={"operation_id": "different"}) != record
