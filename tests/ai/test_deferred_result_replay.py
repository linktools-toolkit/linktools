#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable deferred-result replay."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest
from pydantic_ai.messages import StructuredContent
from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    canonical_json_bytes,
    canonical_sha256,
    ExternalCallStatus,
    Principal,
    ExecutionStatus,
    TenantAuthorizationPolicy,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._approval import (
    DefaultApprovalService,
)
from linktools.ai.runtime._external import (
    DefaultExternalService,
)
from linktools.ai.execution import (
    ApprovalDecisionRequest, ExternalCallSucceeded, ExternalCallRetry, ExternalCallFailed, ExternalSupplyRequest,
    ExecutionRecoveryDetail, ExecutionRecoveryState,
)
from linktools.ai.persistence.contracts import (
    deferred_resource_id,
    ApprovalRecord,
    ExternalCallRecord,
    PendingDeferredCall,
    PendingToolContinuation,
)
from linktools.ai.storage import InMemoryObjectStore, StoredPayload
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from linktools.ai.model._tool_return_codec import encode_tool_return_content
from linktools.ai.core import Principal, PrincipalKind


class _AllowAuthorization:
    async def authorize(
        self,
        principal: object,
        action: object,
        resource: object,
    ) -> None:
        del principal, action, resource


_ValueT = TypeVar("_ValueT")


class _ReadStateStore:
    async def read(self, callback: Callable[[object], Awaitable[_ValueT]]) -> _ValueT:
        return await callback(None)


class _Executions:
    def __init__(self, pending: PendingDeferredCall, *, approvals: bool) -> None:
        self.state_store = _ReadStateStore()
        self.execution = SimpleNamespace(
            execution_id="execution", principal_id="owner",
            status=ExecutionStatus.WAITING_DEFERRED,
            recovery_state=ExecutionRecoveryState.WAITING,
            recovery=ExecutionRecoveryDetail(
                agent_run_id="step",
                pending_tools=PendingToolContinuation(
                    "step",
                    approvals=(pending,) if approvals else (),
                    calls=() if approvals else (pending,),
                ),
            ),
        )

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.execution
        return None


class _ExternalCalls:
    def __init__(self, record: ExternalCallRecord) -> None:
        self.record = record
        self.tenant_id = "tenant"

    async def get(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ExternalCallRecord | None:
        if call_id != self.record.call_id or tenant_id != self.tenant_id:
            return None
        return self.record

    async def get_header(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if call_id != self.record.call_id or tenant_id != self.tenant_id:
            return None
        return ResourceRef(ResourceKind.EXTERNAL_CALL, call_id, tenant_id)

    async def supply(
        self,
        call_id: str,
        *,
        tenant_id: str,
        expected_status: ExternalCallStatus,
        idempotency_key: str,
        resolution_kind: str,
        result_payload: StoredPayload | None,
        resolution_metadata: dict[str, object],
        supplied_at: datetime,
    ) -> ExternalCallRecord:
        if (
            call_id != self.record.call_id
            or tenant_id != self.tenant_id
            or self.record.status is not expected_status
        ):
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        self.record = replace(
            self.record,
            status=ExternalCallStatus.SUPPLIED,
            idempotency_key=idempotency_key,
            supplied_at=supplied_at,
            resolution_kind=resolution_kind,
            result_payload=result_payload,
            resolution_metadata=resolution_metadata,
        )
        return self.record


class _Approvals:
    def __init__(self, record: ApprovalRecord) -> None:
        self.record = record
        self.tenant_id = "tenant"

    async def get(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ApprovalRecord | None:
        if approval_id != self.record.approval_id or tenant_id != self.tenant_id:
            return None
        return self.record

    async def get_header(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if approval_id != self.record.approval_id or tenant_id != self.tenant_id:
            return None
        return ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def list_pending(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ApprovalRecord, ...]:
        if (
            execution_id == self.record.execution_id
            and tenant_id == self.tenant_id
            and self.record.status is ApprovalStatus.PENDING
        ):
            return (self.record,)
        return ()

    async def decide(
        self,
        approval_id: str,
        *,
        tenant_id: str,
        expected_status: ApprovalStatus,
        idempotency_key: str,
        decision: ApprovalDecision,
        principal_id: str,
        decision_digest: str,
        decided_at: datetime,
        decision_message: str | None = None,
        resolution_metadata: dict[str, object] | None = None,
    ) -> ApprovalRecord:
        if (
            approval_id != self.record.approval_id
            or tenant_id != self.tenant_id
            or self.record.status is not expected_status
        ):
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        self.record = replace(
            self.record,
            status=(
                ApprovalStatus.APPROVED
                if decision is ApprovalDecision.APPROVE
                else ApprovalStatus.DENIED
            ),
            idempotency_key=idempotency_key,
            decision=decision,
            decided_by=principal_id,
            decision_digest=decision_digest,
            decided_at=decided_at,
            decision_message=decision_message,
            resolution_metadata=(
                {} if resolution_metadata is None else resolution_metadata
            ),
        )
        return self.record


@pytest.mark.asyncio
async def test_external_supply_exact_replay_uses_durable_result() -> None:
    principal = Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value)
    now = datetime.now(timezone.utc)
    pending = PendingDeferredCall(
        "tool-call",
        "external_tool",
        StoredPayload.inline_json({"value": 1}),
    )
    call_id = deferred_resource_id("external-call-v1", "tenant", "execution", "step", pending.tool_call_id)
    calls = _ExternalCalls(
        ExternalCallRecord(
            call_id=call_id,
            execution_id="execution",
            status=ExternalCallStatus.PENDING,
            idempotency_key=None,
            created_at=now,
            supplied_at=None,
        )
    )
    executions = _Executions(pending, approvals=False)
    service = DefaultExternalService(
        calls,
        executions,
        _AllowAuthorization(),
        objects=InMemoryObjectStore(),
        object_key_factory=object.__new__(ObjectKeyFactory),

    )
    request = ExternalSupplyRequest(
        principal,
        call_id,
        "supply-key",
        ExternalCallSucceeded({"result": "ok"}),
    )

    first = await service.supply("execution", request)
    second = await service.supply("execution", request)

    assert first == second
    assert second.accepted is True
    assert calls.record.result_payload == StoredPayload.inline_json(
        encode_tool_return_content({"result": "ok"})
    )

    conflicting = ExternalSupplyRequest(
        principal,
        call_id,
        "supply-key",
        ExternalCallSucceeded({"result": "different"}),
    )
    with pytest.raises(AIError) as error:
        await service.supply("execution", conflicting)
    assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_approval_exact_replay_requires_same_actor() -> None:
    first_principal = Principal("approver-1", "tenant", "service")
    other_principal = Principal("approver-2", "tenant", "service")
    now = datetime.now(timezone.utc)
    pending = PendingDeferredCall(
        "approval-call",
        "approval_tool",
        StoredPayload.inline_json({"value": 1}),
    )
    approval_id = deferred_resource_id(
        "approval-v1",
        "tenant", "execution", "step", pending.tool_call_id
    )
    approvals = _Approvals(
        ApprovalRecord(
            approval_id=approval_id,
            execution_id="execution",
            status=ApprovalStatus.PENDING,
            idempotency_key=None,
            decision=None,
            decided_by=None,
            decision_digest=None,
            created_at=now,
            decided_at=None,
        )
    )
    executions = _Executions(pending, approvals=True)
    service = DefaultApprovalService(
        approvals,
        executions,
        _AllowAuthorization(),
        objects=InMemoryObjectStore(),
    )

    first = await service.decide(
        "execution",
        ApprovalDecisionRequest(
            first_principal,
            approval_id,
            "approval-key",
            ApprovalDecision.APPROVE,
        ),
    )
    persisted_digest = approvals.record.decision_digest
    second = await service.decide(
        "execution",
        ApprovalDecisionRequest(
            first_principal,
            approval_id,
            "approval-key",
            ApprovalDecision.APPROVE,
        ),
    )

    assert first == second
    assert approvals.record.idempotency_key == "approval-key"
    assert approvals.record.decided_by == first_principal.principal_id
    assert approvals.record.decision_digest == persisted_digest

    with pytest.raises(AIError) as actor_error:
        await service.decide(
            "execution",
            ApprovalDecisionRequest(
                other_principal,
                approval_id,
                "approval-key",
                ApprovalDecision.APPROVE,
            ),
        )
    assert actor_error.value.code is ErrorCode.APPROVAL_CONFLICT
    assert approvals.record.decided_by == first_principal.principal_id
    assert approvals.record.decision_digest == persisted_digest

    conflicting = ApprovalDecisionRequest(
        first_principal,
        approval_id,
        "approval-key",
        ApprovalDecision.DENY,
    )
    with pytest.raises(AIError) as error:
        await service.decide("execution", conflicting)
    assert error.value.code is ErrorCode.APPROVAL_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("approvals", (True, False))
async def test_deferred_list_rejects_another_tenant(approvals: bool) -> None:
    pending = PendingDeferredCall("call", "tool", StoredPayload.inline_json({}))
    executions = _Executions(pending, approvals=approvals)
    if approvals:
        service = DefaultApprovalService(
            object(), executions, TenantAuthorizationPolicy(),
            objects=InMemoryObjectStore(),
        )
    else:
        service = DefaultExternalService(
            object(), executions, TenantAuthorizationPolicy(),
            objects=InMemoryObjectStore(),
            object_key_factory=object.__new__(ObjectKeyFactory),

        )
    with pytest.raises(AIError) as error:
        await service.list("execution", principal=Principal("other", "another-tenant", "service"))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("approvals", (True, False))
async def test_deferred_list_respects_custom_authorization(approvals: bool) -> None:
    class Deny:
        async def authorize(self, principal: Principal, action: object, resource: ResourceRef) -> None:
            assert principal.tenant_id == resource.tenant_id
            assert resource.owner_principal_id is None
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    pending = PendingDeferredCall("call", "tool", StoredPayload.inline_json({}))
    executions = _Executions(pending, approvals=approvals)
    if approvals:
        service = DefaultApprovalService(object(), executions, Deny(), objects=InMemoryObjectStore())
    else:
        service = DefaultExternalService(
            object(), executions, Deny(), objects=InMemoryObjectStore(),
            object_key_factory=object.__new__(ObjectKeyFactory),
        )
    with pytest.raises(AIError) as error:
        await service.list("execution", principal=Principal("approver", "tenant", "service"))
    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


def _structured_external_service(*, inline: bool = True):
    pending = PendingDeferredCall('tool-call', 'external_tool', StoredPayload.inline_json({}))
    call_id = deferred_resource_id('external-call-v1', 'tenant', 'execution', 'step', pending.tool_call_id)
    calls = _ExternalCalls(ExternalCallRecord(
        call_id=call_id, execution_id='execution', status=ExternalCallStatus.PENDING,
        idempotency_key=None, created_at=datetime.now(timezone.utc), supplied_at=None,
    ))
    objects = InMemoryObjectStore()
    factory = ObjectKeyFactory('external-replay')
    service = DefaultExternalService(
        calls, _Executions(pending, approvals=False), _AllowAuthorization(),
        objects=objects, object_key_factory=factory,

    )
    request = ExternalSupplyRequest(
        Principal('workspace', 'tenant', PrincipalKind.LOCAL_TRUSTED.value), call_id, 'supply-key',
        ExternalCallSucceeded(StructuredContent({'count': 1}, content='fixed rendering', tag='state')),
    )
    return service, calls, objects, factory, request


async def _legacy_external_result(objects, factory, *, inline: bool) -> StoredPayload:
    payload = {'contract': 'linktools.tool-return', 'version': 1,
               'value': {'type': 'text-content', 'value': {
                   'kind': 'text-content', 'content': 'fixed rendering', 'metadata': None,
               }}}
    if inline:
        return StoredPayload.inline_json(payload)
    reference = await put_runtime_object(objects, factory, PersistenceDomain.CHECKPOINT, 'tenant',
                                         canonical_json_bytes(payload))
    return StoredPayload.object(reference)


def _supplied_record(record: ExternalCallRecord, payload: StoredPayload) -> ExternalCallRecord:
    return replace(record, status=ExternalCallStatus.SUPPLIED,
                   idempotency_key='supply-key',
                   supplied_at=datetime.now(timezone.utc), resolution_kind='succeeded', result_payload=payload)


@pytest.mark.asyncio
@pytest.mark.parametrize('inline', [True, False], ids=['inline', 'object'])
async def test_external_legacy_result_retry_uses_stored_projection_without_rewriting(inline: bool) -> None:
    service, calls, objects, factory, request = _structured_external_service(inline=not inline)
    payload = await _legacy_external_result(objects, factory, inline=inline)
    calls.record = original = _supplied_record(calls.record, payload)
    assert (await service.supply('execution', request)).accepted
    changed_hidden_value = replace(request, resolution=ExternalCallSucceeded(
        StructuredContent({'count': 2}, content='fixed rendering', tag='changed')))
    assert (await service.supply('execution', changed_hidden_value)).accepted
    assert calls.record is original
    if not inline:
        assert await read_runtime_object(objects, payload.ref) == canonical_json_bytes({
            'contract': 'linktools.tool-return', 'version': 1,
            'value': {'type': 'text-content', 'value': {
                'kind': 'text-content', 'content': 'fixed rendering', 'metadata': None,
            }},
        })
    with pytest.raises(AIError) as error:
        await service.supply('execution', replace(request, resolution=ExternalCallSucceeded(
            StructuredContent({'count': 1}, content='different rendering', tag='state'))))
    assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize('field,value', [('value', {'count': 2}), ('tag', 'changed'),
                                        ('description', 'changed'), ('metadata', {'changed': True})])
async def test_external_pending_first_supply_uses_current_structured_identity(
    field: str, value: object,
) -> None:
    service, calls, objects, factory, request = _structured_external_service()
    first = await service.supply('execution', request)
    original = calls.record
    payload = original.result_payload
    assert payload.kind == "inline"
    assert not payload.object_refs()
    raw = payload.decode()
    assert raw['version'] == 2
    assert (await service.supply('execution', request)) == first
    changed = replace(request.resolution.value, **{field: value})
    with pytest.raises(AIError) as error:
        await service.supply('execution', replace(request, resolution=ExternalCallSucceeded(changed)))
    assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT
    assert calls.record is original


@pytest.mark.asyncio
@pytest.mark.parametrize('inline', [True, False], ids=['inline', 'object'])
@pytest.mark.parametrize('conflict', [ErrorCode.EXTERNAL_RESULT_CONFLICT, ErrorCode.STORAGE_CONFLICT])
async def test_external_concurrent_legacy_winner_selects_retry_projection(inline: bool, conflict: ErrorCode) -> None:
    service, calls, objects, factory, request = _structured_external_service(inline=inline)
    payload = await _legacy_external_result(objects, factory, inline=inline)
    winner = _supplied_record(calls.record, payload)

    async def concurrent_supply(*args, **kwargs):
        calls.record = winner
        raise AIError(conflict)

    calls.supply = concurrent_supply
    assert (await service.supply('execution', request)).accepted
    assert calls.record is winner


@pytest.mark.asyncio
@pytest.mark.parametrize('committed', [True, False])
async def test_external_supply_reads_durable_outcome_after_uncertain_commit(committed: bool) -> None:
    service, calls, objects, factory, request = _structured_external_service()
    supply = calls.supply
    failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async def interrupted_supply(*args, **kwargs):
        if committed:
            await supply(*args, **kwargs)
        raise failure

    calls.supply = interrupted_supply
    if committed:
        assert (await service.supply('execution', request)).accepted
        assert calls.record.status is ExternalCallStatus.SUPPLIED
    else:
        with pytest.raises(type(failure)):
            await service.supply('execution', request)
        assert calls.record.status is ExternalCallStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize('committed', [True, False])
async def test_external_supply_preserves_cancellation_after_durable_readback(committed: bool) -> None:
    service, calls, objects, factory, request = _structured_external_service()
    supply, get = calls.supply, calls.get
    interrupted = asyncio.Event()
    readback = asyncio.Event()

    async def interrupted_supply(*args, **kwargs):
        if committed:
            await supply(*args, **kwargs)
        interrupted.set()
        await asyncio.Event().wait()

    async def observe_readback(*args, **kwargs):
        result = await get(*args, **kwargs)
        if interrupted.is_set():
            readback.set()
        return result

    calls.supply, calls.get = interrupted_supply, observe_readback
    task = asyncio.create_task(service.supply('execution', request))
    await asyncio.wait_for(interrupted.wait(), timeout=5)
    task.cancel('caller stopped')
    with pytest.raises(asyncio.CancelledError, match='caller stopped'):
        await task
    assert task.cancelled()
    assert readback.is_set()
    assert calls.record.status is (ExternalCallStatus.SUPPLIED if committed else ExternalCallStatus.PENDING)
    calls.supply = supply
    assert (await service.supply('execution', request)).accepted


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [
    RuntimeError('programming failure'), AIError(ErrorCode.AUTHORIZATION_DENIED), AIError(ErrorCode.STORAGE_UNAVAILABLE),
])
async def test_external_supply_does_not_hide_unclassified_failures_after_commit(failure: Exception) -> None:
    service, calls, objects, factory, request = _structured_external_service()
    supply = calls.supply

    async def failing_supply(*args, **kwargs):
        await supply(*args, **kwargs)
        raise failure

    calls.supply = failing_supply
    with pytest.raises(type(failure)) as error:
        await service.supply('execution', request)
    assert error.value is failure
    assert calls.record.status is ExternalCallStatus.SUPPLIED
    assert (await service.supply('execution', request)).accepted


@pytest.mark.asyncio
@pytest.mark.parametrize('resolution_type', [ExternalCallRetry, ExternalCallFailed])
async def test_external_non_success_replay_retains_message_and_kind(
    resolution_type: type[ExternalCallRetry] | type[ExternalCallFailed],
) -> None:
    service, calls, objects, factory, request = _structured_external_service()
    request = replace(request, resolution=resolution_type('message'))
    first = await service.supply('execution', request)
    assert await service.supply('execution', request) == first
    for changed in [resolution_type('different'), ExternalCallSucceeded('message')]:
        with pytest.raises(AIError) as error:
            await service.supply('execution', replace(request, resolution=changed))
        assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_external_retry_verifies_original_object_before_selecting_projection() -> None:
    service, calls, objects, factory, request = _structured_external_service()
    payload = await _legacy_external_result(objects, factory, inline=False)
    calls.record = _supplied_record(calls.record, payload)
    objects._objects[payload.ref.key] = b'{"version":2}'
    with pytest.raises(AIError) as error:
        await service.supply('execution', request)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_external_retry_rejects_unknown_stored_result_version() -> None:
    service, calls, objects, factory, request = _structured_external_service()
    payload = encode_tool_return_content(request.resolution.value)
    payload['version'] = 3
    calls.record = _supplied_record(calls.record, StoredPayload.inline_json(payload))
    with pytest.raises(AIError) as error:
        await service.supply('execution', request)
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', [False, True], ids=['same-result', 'changed-value'])
async def test_external_independent_sqlite_contenders_share_one_result_owner(
    tmp_path: Path, changed: bool,
) -> None:
    from linktools.ai.persistence import Persistence, PersistenceStorage

    _, seed, objects, factory, request = _structured_external_service()
    first = Persistence(PersistenceStorage.sqlite(tmp_path / 'external.db'))
    second = Persistence(PersistenceStorage.sqlite(tmp_path / 'external.db'))
    await first.initialize(namespace='external-race', tenant_id='tenant')
    await second.initialize(namespace='external-race', tenant_id='tenant')
    try:
        await first.checkpoint.external_calls.create_call(seed.record)
        ready = asyncio.Event()
        count = 0

        class Contender:
            def __init__(self, repository) -> None:
                self.repository = repository
                self.read = False

            async def get(self, *args, **kwargs):
                nonlocal count
                value = await self.repository.get(*args, **kwargs)
                if not self.read:
                    self.read = True
                    count += 1
                    if count == 2:
                        ready.set()
                    await ready.wait()
                return value

            async def supply(self, *args, **kwargs):
                return await self.repository.supply(*args, **kwargs)

        pending = PendingDeferredCall('tool-call', 'external_tool', StoredPayload.inline_json({}))
        services = [DefaultExternalService(
            Contender(state.checkpoint.external_calls), _Executions(pending, approvals=False), _AllowAuthorization(),
            objects=objects, object_key_factory=factory,
        ) for state in (first, second)]
        other = replace(request, resolution=ExternalCallSucceeded(
            StructuredContent({'count': 2 if changed else 1}, content='fixed rendering', tag='state')))
        results = await asyncio.wait_for(asyncio.gather(
            services[0].supply('execution', request), services[1].supply('execution', other),
            return_exceptions=True,
        ), timeout=10)
        if changed:
            assert sum(isinstance(result, AIError) for result in results) == 1
            assert next(result for result in results if isinstance(result, AIError)).code is ErrorCode.EXTERNAL_RESULT_CONFLICT
        else:
            assert all(result.accepted for result in results)
        winner = await first.checkpoint.external_calls.get(request.call_id, tenant_id='tenant')
        assert winner.status is ExternalCallStatus.SUPPLIED
        assert winner.result_payload.decode()['version'] == 2
    finally:
        await first.close()
        await second.close()
