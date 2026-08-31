#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contract tests for ToolRepository optimistic-CAS retry wiring."""

from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import ToolOperationStatus, canonical_sha256
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state import ToolOperationAdmission
from linktools.ai.runtime.state._repositories import ToolRepositoryImpl
from linktools.ai.storage import StoredPayload


def _record() -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="tool-operation",
        tenant_id="tenant",
        step_run_id="step-run",
        tool_call_id="tool-call",
        idempotency_key_digest=canonical_sha256({"call": "tool-call"}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        status=ToolOperationStatus.CLAIMED,
        owner="tool-owner",
        fence=1,
        lease_expires_at=now + timedelta(seconds=60),
        error_code=None,
        created_at=now,
        updated_at=now,
    )


def _admission() -> ToolOperationAdmission:
    return ToolOperationAdmission(
        tenant_id="tenant",
        tool_operation_id="tool-operation",
        step_run_id="step-run",
        recovery_step_run_id=None,
        tool_call_id="tool-call",
        idempotency_key_digest=canonical_sha256({"call": "tool-call"}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        owner="tool-owner",
        lease_seconds=60,
    )


async def _invoke_retry_method(
    repository: ToolRepositoryImpl,
    method_name: str,
) -> ToolOperationRecord:
    if method_name == "admit":
        return await repository.admit(_admission())
    if method_name == "reserve":
        return await repository.reserve(_record())
    if method_name == "claim":
        return await repository.claim(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            lease_seconds=60,
        )
    if method_name == "complete_payload":
        return await repository.complete_payload(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            result_payload=StoredPayload.inline_bytes(b"result"),
        )
    if method_name == "fail":
        return await repository.fail(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            error_code="tool_failed",
        )
    if method_name == "fail_payload":
        return await repository.fail_payload(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            error_code="tool_failed",
            error_payload=StoredPayload.inline_bytes(b"error"),
        )
    if method_name == "mark_effect_unknown":
        return await repository.mark_effect_unknown(
            "tool-operation",
            tenant_id="tenant",
            owner="tool-owner",
            fence=1,
            error_code="effect_unknown",
        )
    raise AssertionError(f"unexpected retry method: {method_name}")


@pytest.mark.parametrize(
    "method_name",
    (
        "admit",
        "reserve",
        "claim",
        "complete_payload",
        "fail",
        "fail_payload",
        "mark_effect_unknown",
    ),
)
@pytest.mark.asyncio
async def test_tool_mutations_use_raw_storage_conflict_retry(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    repository = object.__new__(ToolRepositoryImpl)
    expected = _record()
    calls = 0

    async def retry(self, operation):
        nonlocal calls
        del self, operation
        calls += 1
        return expected

    monkeypatch.setattr(ToolRepositoryImpl, "_retry_storage_conflict", retry)

    result = await _invoke_retry_method(repository, method_name)

    assert result is expected
    assert calls == 1


class _RenewStore:
    def __init__(self, result: ToolOperationRecord) -> None:
        self.result = result
        self.calls = 0

    async def mutate(self, operation) -> ToolOperationRecord:
        del operation
        self.calls += 1
        return self.result


@pytest.mark.asyncio
async def test_tool_renew_does_not_use_raw_storage_conflict_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = object.__new__(ToolRepositoryImpl)
    expected = _record()
    store = _RenewStore(expected)
    repository._tenant_id = "tenant"
    repository._store = store

    async def unexpected_retry(self, operation):
        del self, operation
        raise AssertionError("renew must not use raw STORAGE_CONFLICT retry")

    monkeypatch.setattr(ToolRepositoryImpl, "_retry_storage_conflict", unexpected_retry)

    result = await repository.renew(
        "tool-operation",
        tenant_id="tenant",
        owner="tool-owner",
        fence=1,
        lease_seconds=60,
    )

    assert result is expected
    assert store.calls == 1
