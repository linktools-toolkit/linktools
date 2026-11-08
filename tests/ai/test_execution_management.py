#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution management validates durable truth before exposing control results."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import (
    AuthorizationAction, ExecutionStatus, IdempotencyStatus, Principal,
    ResourceKind, ResourceRef, StopReason, TenantAuthorizationPolicy, UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionManager, IdempotencyRecord, ResultRecord
from linktools.ai.storage import StoredPayload

from .test_runtime_persistence_evidence import _execution

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("principal", "tenant", "service")


class _ReadPort:
    def __init__(self, execution, identity=None, header_owner="principal") -> None:
        self.execution = execution
        self.identity = identity
        self.header_owner = header_owner
        self.reads = []

    async def read_execution(self, execution_id, *, tenant_id):
        self.reads.append(("execution", execution_id, tenant_id))
        return self.execution

    async def read_source_execution(self, execution_id, *, tenant_id):
        self.reads.append(("source_execution", execution_id, tenant_id))
        return self.execution

    async def read_source_header(self, execution_id, *, tenant_id):
        self.reads.append(("header", execution_id, tenant_id))
        return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, self.header_owner)

    async def read_header(self, execution_id, *, tenant_id):
        return await self.read_source_header(execution_id, tenant_id=tenant_id)

    async def get_idempotency(self, scope, key_digest, *, tenant_id):
        return self.identity

    async def result_record(self, execution, *, tenant_id):
        return execution.result


@pytest.mark.parametrize("status,output,error", (
    (ExecutionStatus.SUCCEEDED, None, None),
    (ExecutionStatus.CANCELLED, StoredPayload.inline_json("unexpected"), ErrorCode.EXECUTION_CANCELLED.value),
    (ExecutionStatus.FAILED, None, None),
))
async def test_results_reject_inconsistent_terminal_truth(status, output, error) -> None:
    record = replace(_execution(), status=status, error_code=error,
                     result=ResultRecord(output, StopReason.CANCELLED, UsageMetrics(), datetime.now(timezone.utc)))
    manager = ExecutionManager(_ReadPort(record), TenantAuthorizationPolicy())
    with pytest.raises(AIError) as caught:
        await manager.result_from_execution(record, tenant_id="tenant")
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_admission_does_not_reinterpret_unknown_commit_as_started() -> None:
    now = datetime.now(timezone.utc)
    identity = IdempotencyRecord("execution.run", "key", "request", ResourceKind.EXECUTION,
        "execution", IdempotencyStatus.START_UNKNOWN, None, None, now, now)
    manager = ExecutionManager(_ReadPort(_execution(), identity), TenantAuthorizationPolicy())
    with pytest.raises(AIError) as caught:
        await manager.resolve_admission("execution.run", "key", "request", _PRINCIPAL)
    assert caught.value.code is ErrorCode.EXECUTION_START_UNKNOWN


async def test_custom_authorization_refreshes_loaded_record_and_rechecks_each_operation() -> None:
    record = replace(_execution(), principal_id="winner")
    refreshed = replace(record, revision=record.revision + 1)
    port = _ReadPort(record, header_owner="unrelated-header")
    observations = []
    allowed = True

    class Authorization:
        async def authorize(self, principal, action, resource):
            observations.append((action, resource.owner_principal_id))
            if not allowed:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            port.execution = refreshed

    manager = ExecutionManager(port, Authorization())
    assert await manager.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ) is refreshed
    assert port.reads == [("execution", "execution", "tenant")] * 2
    allowed = False
    with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
        await manager.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ)
    assert observations == [(AuthorizationAction.EXECUTION_READ, "winner")] * 2
    assert port.reads == [("execution", "execution", "tenant")] * 3


async def test_default_authorization_reuses_source_record_and_rechecks_each_operation() -> None:
    record = _execution()
    port = _ReadPort(record, header_owner="unrelated-header")
    manager = ExecutionManager(port, TenantAuthorizationPolicy())
    assert await manager.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ) is record
    assert port.reads == [("execution", "execution", "tenant")]
    port.execution = replace(record, principal_id="another-owner")
    with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
        await manager.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ)
    assert port.reads == [("execution", "execution", "tenant")] * 2


async def test_changed_loaded_owner_is_authorized_without_unrelated_header_read() -> None:
    record = replace(_execution(), principal_id="winner")
    port = _ReadPort(record, header_owner="principal")
    manager = ExecutionManager(port, TenantAuthorizationPolicy())
    with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
        await manager.authorize_loaded_execution(
            record, _PRINCIPAL, AuthorizationAction.EXECUTION_READ,
            owner_principal_id="observed",
        )
    assert port.reads == []


async def test_admission_authorizes_its_one_source_record() -> None:
    now = datetime.now(timezone.utc)
    identity = IdempotencyRecord("execution.run", "key", "request", ResourceKind.EXECUTION,
        "execution", IdempotencyStatus.STARTED, None, None, now, now)
    record = _execution()
    port = _ReadPort(record, identity)
    manager = ExecutionManager(port, TenantAuthorizationPolicy())
    assert await manager.resolve_admission("execution.run", "key", "request", _PRINCIPAL) == (identity, record)
    assert port.reads == [("source_execution", "execution", "tenant")]


async def test_remote_cancel_requires_authorization_for_changed_pending_owner() -> None:
    from linktools.ai.execution import CancelExecutionRequest

    observed = replace(_execution(), status=ExecutionStatus.STARTED, principal_id="observed")
    winner = replace(observed, principal_id="winner")

    class PendingPort(_ReadPort):
        signalled = False

        async def read_header(self, execution_id, *, tenant_id):
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id, "observed")

        def has_local_pending(self, execution_id, *, tenant_id):
            return False

        async def read_source_execution(self, execution_id, *, tenant_id):
            return None

        async def read_remote_pending(self, execution_id, *, tenant_id):
            return winner

        async def signal_remote_cancel(self, execution_id, *, tenant_id):
            self.signalled = True

    class Authorization:
        async def authorize(self, principal, action, resource):
            if resource.owner_principal_id == "winner":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    port = PendingPort(observed, header_owner="winner")
    manager = ExecutionManager(port, Authorization())
    with pytest.raises(AIError) as caught:
        await manager.cancel_authorized("execution", CancelExecutionRequest(_PRINCIPAL, "cancel"))
    assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert not port.signalled


async def test_cancel_authorization_reuses_its_source_record() -> None:
    from linktools.ai.execution import CancelExecutionRequest

    class SourcePort(_ReadPort):
        def has_local_pending(self, execution_id, *, tenant_id):
            return False

    record = _execution()
    port = SourcePort(record)
    manager = ExecutionManager(port, TenantAuthorizationPolicy())
    assert await manager._load_cancel_execution(
        "execution", CancelExecutionRequest(_PRINCIPAL, "cancel"),
    ) is record
    assert port.reads == [("source_execution", "execution", "tenant")]


@pytest.mark.parametrize("mismatch,expected_error", (
    (None, None),
    ("request", ErrorCode.IDEMPOTENCY_CONFLICT),
    ("scope", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("key", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("kind", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("owner", ErrorCode.AUTHORIZATION_DENIED),
))
async def test_admission_reuses_observed_identity_without_bypassing_validation(
    mismatch: str | None, expected_error: ErrorCode | None,
) -> None:
    now = datetime.now(timezone.utc)
    identity = IdempotencyRecord("execution.run", "key", "request", ResourceKind.EXECUTION,
        "execution", IdempotencyStatus.STARTED, None, None, now, now)
    record = _execution()
    if mismatch == "request":
        identity = replace(identity, request_digest="other")
    elif mismatch == "scope":
        identity = replace(identity, scope="other")
    elif mismatch == "key":
        identity = replace(identity, idempotency_key_digest="other")
    elif mismatch == "kind":
        identity = replace(identity, resource_kind=ResourceKind.SESSION)
    elif mismatch == "owner":
        record = replace(record, principal_id="other")

    class ObservedPort(_ReadPort):
        async def get_idempotency(self, scope, key_digest, *, tenant_id):
            raise AssertionError("Admission must reuse the observed source identity")

    port = ObservedPort(record)
    manager = ExecutionManager(port, TenantAuthorizationPolicy())
    if expected_error is None:
        assert await manager.resolve_admission(
            "execution.run", "key", "request", _PRINCIPAL, identity=identity,
        ) == (identity, record)
    else:
        with pytest.raises(AIError) as caught:
            await manager.resolve_admission(
                "execution.run", "key", "request", _PRINCIPAL, identity=identity,
            )
        assert caught.value.code is expected_error
    assert port.reads == (
        [("source_execution", "execution", "tenant")] if mismatch in {None, "owner"} else []
    )
