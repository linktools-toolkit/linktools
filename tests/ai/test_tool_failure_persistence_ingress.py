#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool failure payload errors keep their typed persistence semantics."""

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

import pytest
from linktools.ai.core import JsonValue, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
)
from linktools.ai.runtime.state._repository_common import (
    _restore_lease_fields,
    domain_data as _domain_data,
)
from linktools.ai.storage import StoredPayload


def _record() -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="b" * 64,
        tool_name="tool",
        arguments_digest="c" * 64,
        binding_digest="a" * 64,
        replay_safe=True,
        status=ToolOperationStatus.FAILED,
        owner=None,
        fence=1,
        lease_expires_at=None,
        error_code=ErrorCode.TOOL_RETRY_REQUIRED.value,
        created_at=now,
        updated_at=now,
        error_payload=StoredPayload.inline_bytes(
            b'{"kind":"tool_call_rejected","message":"retry","version":1}'
        ),
    )


def _decode_with_fields(**overrides: object) -> ToolOperationRecord:
    encoded_overrides = {
        name: _encode_persisted_domain(cast(Any, value))
        for name, value in overrides.items()
    }

    def replace_fields(value: JsonValue) -> JsonValue:
        restored = _restore_lease_fields(value, ToolOperationRecord)
        assert isinstance(restored, Mapping)
        fields = restored.get("fields")
        assert isinstance(fields, Mapping)
        return cast(
            JsonValue,
            {
                **restored,
                "fields": {
                    **fields,
                    **encoded_overrides,
                },
            },
        )

    return _decode_enveloped_domain(
        cast("Mapping[str, JsonValue]", _domain_data(_record())),
        ToolOperationRecord,
        payload_transform=replace_fields,
    )


def _decode_with_error_payload(payload: StoredPayload) -> ToolOperationRecord:
    return _decode_with_fields(error_payload=payload)


@pytest.mark.parametrize(
    "value",
    (
        {"kind": "tool_call_rejected", "message": "retry"},
        {"version": 2, "kind": "tool_call_rejected", "message": "retry"},
    ),
)
def test_persisted_tool_failure_rejects_unsupported_payload_version(
    value: dict[str, object],
) -> None:
    with pytest.raises(AIError) as captured:
        _decode_with_error_payload(
            StoredPayload.inline_bytes(json.dumps(value).encode("utf-8"))
        )
    assert captured.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.parametrize(
    "value",
    (
        {
            "version": 1,
            "kind": "tool_call_rejected",
            "message": "retry",
            "extra": 1,
        },
        {"version": 1, "kind": "tool_call_failed", "message": "failed"},
    ),
)
def test_persisted_tool_failure_rejects_corrupt_current_payload(
    value: dict[str, object],
) -> None:
    with pytest.raises(AIError) as captured:
        _decode_with_error_payload(
            StoredPayload.inline_bytes(json.dumps(value).encode("utf-8"))
        )
    assert captured.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    "overrides",
    (
        {"result_payload": StoredPayload.inline_bytes(b"result")},
        {
            "status": ToolOperationStatus.COMPLETED,
            "result_payload": StoredPayload.inline_bytes(b"result"),
        },
        {"status": ToolOperationStatus.PENDING},
        {"status": ToolOperationStatus.CANCELLED},
        {
            "status": ToolOperationStatus.EFFECT_UNKNOWN,
            "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value,
        },
    ),
)
def test_persisted_tool_operation_rejects_conflicting_state_payloads(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(AIError) as captured:
        _decode_with_fields(**overrides)
    assert captured.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
