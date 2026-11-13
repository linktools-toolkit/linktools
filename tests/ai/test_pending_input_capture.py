#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input capture requires authorized promotion and a fresh source record."""

from linktools.ai.persistence import Persistence
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    AuthorizationAction,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecord
from linktools.ai.inputs import CaptureInputRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.inputs import InputCaptures
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from .test_history_projection_conformance import _record

_PRINCIPAL = Principal("principal", "tenant", "service")


@asynccontextmanager
async def _storage():
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="pending-capture", tenant_id="tenant")
    try:
        yield state
    finally:
        await state.close()


def _captures(state, *, prepare=None, authorization=None):
    return InputCaptures.from_persistence(
        state, authorization=authorization or TenantAuthorizationPolicy("tenant"),
        execution=SimpleNamespace(), prepare_pending_capture=prepare,
    )


def _stale_source_miss(monkeypatch: pytest.MonkeyPatch, state: Persistence) -> list[str]:
    repository = state.execution.executions
    store = repository.state_store
    get = repository.get
    source_reads: list[str] = []

    async def observed_get(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
        scope = active_state_scope()
        if scope is None:
            return None
        assert scope.group is store.storage_group
        assert not scope.writable and store in scope.transactions
        source_reads.append(execution_id)
        return await get(execution_id, tenant_id=tenant_id)

    monkeypatch.setattr(repository, "get", observed_get)
    return source_reads


@pytest.mark.asyncio
async def test_pending_capture_authorizes_promotion_and_refreshes_source_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _storage() as state:
        record = _record(ExecutionStatus.STARTED, 1)
        policy = TenantAuthorizationPolicy("tenant")
        promotions = []

        async def prepare(execution_id: str, principal: Principal) -> None:
            header = ResourceRef(ResourceKind.EXECUTION, execution_id, "tenant", record.principal_id)
            await policy.authorize(principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT, header)
            await policy.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
            await state.execution.executions.create_execution(record)
            promotions.append(execution_id)
            assert await state.execution.executions.get(execution_id, tenant_id=principal.tenant_id) is None

        captures = _captures(state, prepare=prepare)
        source_reads = _stale_source_miss(monkeypatch, state)
        reference = await captures.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "capture"))
        captured = await captures.read_agent(reference, principal=_PRINCIPAL)
        assert captured.prompt == "prompt"
        assert captured.source_execution_id == "execution"
        assert captured.source_principal["principal_id"] == "principal"
        assert promotions == ["execution"]
        assert source_reads == ["execution"]


@pytest.mark.asyncio
@pytest.mark.parametrize("principal", (
    Principal("other", "tenant", "service"),
    Principal("principal", "other", "service"),
    Principal("principal", "tenant", "local_trusted"),
))
async def test_denied_pending_capture_does_not_promote(principal) -> None:
    async with _storage() as state:
        record = _record(ExecutionStatus.STARTED, 1)
        policy = TenantAuthorizationPolicy("tenant")
        promotions = []

        async def prepare(execution_id, principal):
            header = ResourceRef(ResourceKind.EXECUTION, execution_id, "tenant", record.principal_id)
            await policy.authorize(principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT, header)
            await policy.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
            promotions.append(execution_id)
            await state.execution.executions.create_execution(record)

        captures = _captures(state, prepare=prepare)
        with pytest.raises(AIError) as raised:
            await captures.capture_input("execution", CaptureInputRequest(principal, "denied"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert promotions == []
        assert await state.execution.executions.get("execution", tenant_id="tenant") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", (AuthorizationAction.EXECUTION_CAPTURE_INPUT, AuthorizationAction.EXECUTION_READ))
async def test_promoted_capture_still_applies_both_canonical_authorization_actions(action) -> None:
    async with _storage() as state:
        record = _record(ExecutionStatus.STARTED, 1)
        checked = []

        class Denied:
            async def authorize(self, principal, requested, header):
                checked.append(requested)
                if requested is action:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        async def prepare(execution_id, principal):
            await state.execution.executions.create_execution(record)

        captures = _captures(state, prepare=prepare, authorization=Denied())
        with pytest.raises(AIError) as raised:
            await captures.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "denied"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert action in checked


@pytest.mark.asyncio
async def test_promotion_callback_does_not_bypass_source_input_owner() -> None:
    async with _storage() as state:
        async def prepare(execution_id, principal):
            await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))

        captures = _captures(state, prepare=prepare)
        with pytest.raises(AIError) as raised:
            await captures.capture_input("execution", CaptureInputRequest(Principal("other", "tenant", "service"), "other"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("prepare_available", (False, True))
async def test_pending_capture_requires_source_header_even_after_preparation(prepare_available) -> None:
    async with _storage() as state:
        async def prepare(execution_id, principal):
            return None

        captures = _captures(state, prepare=prepare if prepare_available else None)
        with pytest.raises(AIError) as raised:
            await captures.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "missing"))
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_source_capture_does_not_request_pending_promotion() -> None:
    async with _storage() as state:
        await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))

        async def prepare(execution_id, principal):
            raise AssertionError("source execution must not request pending promotion")

        captures = _captures(state, prepare=prepare)
        reference = await captures.capture_input("execution", CaptureInputRequest(_PRINCIPAL, "source"))
        assert (await captures.read_agent(reference, principal=_PRINCIPAL)).prompt == "prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("context_policy", ("captured", "clean"))
async def test_pending_capture_reads_remote_promotion_despite_cached_sqlite_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context_policy: str,
) -> None:
    monkeypatch.setattr(_write_cache, "time", lambda: 1000.0)
    path = tmp_path / "pending.db"
    cached = PersistenceStorage.sqlite(path, cache=InMemoryCacheBackend(max_bytes=1_000_000))
    async with Persistence.open("pending-capture", storage=cached, tenant_id="tenant") as reader:
        async with Persistence.open(
            "pending-capture", storage=PersistenceStorage.sqlite(path), tenant_id="tenant",
        ) as writer:
            repository = reader.execution.executions
            assert await repository.get("execution", tenant_id="tenant") is None
            assert await repository.get_header("execution", tenant_id="tenant") is None
            record = _record(ExecutionStatus.STARTED, 1)
            policy = TenantAuthorizationPolicy("tenant")
            promotions: list[str] = []

            async def prepare(execution_id: str, principal: Principal) -> None:
                header = ResourceRef(ResourceKind.EXECUTION, execution_id, "tenant", record.principal_id)
                await policy.authorize(principal, AuthorizationAction.EXECUTION_CAPTURE_INPUT, header)
                await policy.authorize(principal, AuthorizationAction.EXECUTION_READ, header)
                await writer.execution.executions.create_execution(record)
                promotions.append(execution_id)
                assert await writer.execution.executions.get(execution_id, tenant_id="tenant") == record
                assert await repository.get(execution_id, tenant_id="tenant") is None
                assert await repository.get_header(execution_id, tenant_id="tenant") is None

            captures = _captures(reader, prepare=prepare)
            reference = await captures.capture_input("execution", CaptureInputRequest(
                _PRINCIPAL, "remote-promotion", context_policy=context_policy,
            ))
            captured = await captures.read_agent(reference, principal=_PRINCIPAL)
            assert captured.prompt == "prompt"
            assert captured.source_execution_id == "execution"
            assert captured.source_principal["principal_id"] == "principal"
            assert promotions == ["execution"]
