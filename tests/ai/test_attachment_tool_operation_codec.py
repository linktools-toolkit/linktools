#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone

from linktools.ai.core import ToolOperationStatus
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    AttachmentResult,
    ContentRef,
    RuntimeDomain,
    iter_runtime_object_refs,
    managed_attachment_path,
)
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.storage import ObjectRef, StoredPayload


def _record(*, attachment_result: AttachmentResult | None = None) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="a" * 64,
        tool_name="read_attachment" if attachment_result is not None else "read_file",
        arguments_digest="b" * 64,
        binding_digest="c" * 64,
        replay_safe=True,
        status=(
            ToolOperationStatus.COMPLETED
            if attachment_result is not None
            else ToolOperationStatus.CLAIMED
        ),
        owner="owner",
        fence=1,
        lease_expires_at=None,
        error_code=None,
        created_at=now,
        updated_at=now,
        result_payload=(
            StoredPayload.inline_json({"status": "read"})
            if attachment_result is not None
            else None
        ),
        attachment_result=attachment_result,
    )


def _attachment_result() -> AttachmentResult:
    reference = ObjectRef(
        "memory",
        "attachment-body",
        "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881",
        1,
    )
    return AttachmentResult(
        1,
        AttachmentEntry(
            managed_attachment_path("e", "d" * 64, 0),
            "evidence.txt",
            "text/plain",
            AttachmentPresentation(None, None),
            ContentRef(RuntimeDomain.EXECUTION.value, None, reference),
        ),
    )


def test_tool_operation_without_attachment_result_keeps_old_canonical_wire() -> None:
    record = _record()

    encoded = encode_domain(record)

    assert isinstance(encoded, dict)
    fields = encoded["fields"]
    assert isinstance(fields, dict)
    assert "attachment_result" not in fields
    assert decode_domain(encoded, ToolOperationRecord) == record


def test_tool_operation_attachment_result_round_trips_and_keeps_body_reachable() -> None:
    attachment_result = _attachment_result()
    record = _record(attachment_result=attachment_result)

    encoded = encode_domain(record)

    assert decode_domain(encoded, ToolOperationRecord) == record
    refs = tuple(
        iter_runtime_object_refs(
            encoded,
            default_domain=RuntimeDomain.RECOVERY,
        )
    )
    assert refs == ((RuntimeDomain.EXECUTION, attachment_result.entry.content.object),)
