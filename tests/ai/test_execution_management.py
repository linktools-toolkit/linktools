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

    async def read_execution(self, execution_id, *, tenant_id):
        return self.execution

    async def read_source_header(self, execution_id, *, tenant_id):
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


async def test_authorization_rechecks_changed_execution_owner() -> None:
    record = replace(_execution(), principal_id="winner")
    port = _ReadPort(record, header_owner="observed")
    observations = []

    class Authorization:
        async def authorize(self, principal, action, resource):
            observations.append((action, resource.owner_principal_id))
            if len(observations) == 1:
                port.header_owner = "winner"

    manager = ExecutionManager(port, Authorization())
    assert await manager.load_authorized("execution", _PRINCIPAL, AuthorizationAction.EXECUTION_READ) is record
    assert observations == [(AuthorizationAction.EXECUTION_READ, "observed"),
                            (AuthorizationAction.EXECUTION_READ, "winner")]


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
