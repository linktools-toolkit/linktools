#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result reads reuse authorized inline metadata without weakening handoff fences."""

from linktools.ai.persistence import Persistence
import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.runtime._execution import DefaultExecutionService, _ExecutionRuntimeBridge
from linktools.ai.execution.events import LiveExecutionEventBroker
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.storage import ObjectRef, StoredPayload

from .test_incremental_projection_boundaries import _storage
from .test_runtime_persistence_evidence import _execution, _result

pytestmark = pytest.mark.asyncio
_PRINCIPAL = Principal("principal", "tenant", "service")


def _service(state, *, repository=None, authorization=None, objects=None, release=None, history=False):
    if history:
        return History(
            SimpleNamespace(), tenant_id="tenant",
            executions=repository or state.execution.executions,
            authorization=authorization or TenantAuthorizationPolicy(),
            execution_objects=objects or state.object_store(PersistenceDomain.EXECUTION),
        )
    return DefaultExecutionService(
        replace(state.execution, executions=repository or state.execution.executions),
        objects or state.object_store(PersistenceDomain.EXECUTION),
        authorization or TenantAuthorizationPolicy(), sessions=state.conversation.sessions,
        catalog=None, compiler=None, runtime_bridge=_ExecutionRuntimeBridge(),
        live_broker=LiveExecutionEventBroker(), history_reader=None, release_terminal=release,
    )


async def _seed(state, *, output=None):
    execution = _execution()
    result = _result(execution.execution_id, "inline", execution.created_at)
    if output is not None:
        result = replace(result, output=output)
    execution = replace(execution, status=ExecutionStatus.SUCCEEDED, result=result)
    await state.execution.executions.create_execution(execution)
    return execution


@pytest.mark.parametrize("backend", ("memory", "sqlite",))
@pytest.mark.parametrize("history,method", (
    (False, "result"), (False, "result_payload_size"), (False, "wait"), (True, "result"),
))
async def test_terminal_result_queries_use_one_authorized_result_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, history: bool, method: str,
) -> None:
    state = _storage(backend, tmp_path)
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        repository = state.execution.executions
        get = repository.get
        reads = 0
        statements = []

        async def tracked_get(*args, **kwargs):
            nonlocal reads
            reads += 1
            return await get(*args, **kwargs)

        monkeypatch.setattr(repository, "get", tracked_get)
        engine = repository.state_store.storage_group.context.engine.sync_engine if backend == "sqlite" else None

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        if engine is not None:
            event.listen(engine, "before_cursor_execute", capture)
        try:
            value = await getattr(_service(state, history=history), method)("execution", principal=_PRINCIPAL)
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        assert value.output == {"text": "result"} if method in {"result", "wait"} else value == record.result.output.size
        # The default policy never suspends, so Runtime reuses its authorized source record.
        limit = 2 if history else 1
        assert reads <= limit
        assert sum(item.lstrip().upper().startswith(("SELECT", "WITH")) for item in statements) <= limit
    finally:
        await state.close()


@pytest.mark.parametrize("history,method", (
    (False, "result"), (False, "result_payload_size"), (False, "wait"), (True, "result"),
))
async def test_result_metadata_is_read_after_awaited_authorization(history: bool, method: str) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        output = StoredPayload.inline_json({"text": "changed after authorization"})

        class Authorization:
            async def authorize(self, principal, action, resource):
                await asyncio.sleep(0)
                await state.execution.executions.compare_and_swap(
                    "execution", tenant_id="tenant", expected_revision=record.revision,
                    next_record=replace(record, revision=record.revision + 1,
                                        result=replace(record.result, output=output)),
                )

        value = await getattr(_service(state, authorization=Authorization(), history=history), method)(
            "execution", principal=_PRINCIPAL,
        )
        assert value.output == {"text": "changed after authorization"} if method in {"result", "wait"} else value == output.size
    finally:
        await state.close()


@pytest.mark.parametrize("adapter", ("independent", "subclass"))
@pytest.mark.parametrize("history,method", (
    (False, "result"), (False, "result_payload_size"), (False, "wait"), (True, "result"),
))
async def test_custom_result_repository_remains_authoritative(adapter: str, history: bool, method: str) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        repository = state.execution.executions
        output = StoredPayload.inline_json({"source": "custom result owner"})
        calls = []

        class CustomResult:
            async def get_result(self, execution_id, *, tenant_id):
                calls.append((execution_id, tenant_id))
                return replace(record.result, output=output)

        class Independent(CustomResult):
            state_store = repository.state_store
            get = repository.get
            get_header = repository.get_header

        class Derived(CustomResult, ExecutionRepositoryImpl):
            pass

        selected = (Independent() if adapter == "independent" else
                    Derived(repository.state_store, namespace="result-read", tenant_id="tenant"))
        value = await getattr(_service(state, repository=selected, history=history), method)("execution", principal=_PRINCIPAL)
        assert value.output == {"source": "custom result owner"} if method in {"result", "wait"} else value == output.size
        assert calls == [("execution", "tenant")]
    finally:
        await state.close()


@pytest.mark.parametrize("method", ("result", "wait"))
@pytest.mark.parametrize("change", ("missing", "nonterminal", "cancel"))
async def test_result_body_wait_preserves_fresh_handoff_and_consumer_cleanup(
    monkeypatch: pytest.MonkeyPatch, change: str, method: str,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    entered, release = asyncio.Event(), asyncio.Event()
    body = b'{"answer":"canonical"}'
    ref = ObjectRef("result-objects", "result", hashlib.sha256(body).hexdigest(), len(body))
    callbacks = []

    class Objects:
        store_id = ref.store_id

        async def open(self, key):
            assert key == ref.key
            entered.set()
            await release.wait()
            yield body

    async def terminal(execution_id, *, tenant_id):
        callbacks.append((execution_id, tenant_id))
        return True

    try:
        record = await _seed(state, output=StoredPayload.object(ref))
        service = _service(state, objects=Objects(), release=terminal)
        query = asyncio.create_task(getattr(service, method)("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), 2)
        if change == "cancel":
            query.cancel()
            with pytest.raises(asyncio.CancelledError):
                await query
        else:
            async def current(*args, **kwargs):
                return None if change == "missing" else replace(record, status=ExecutionStatus.STARTED, result=None)

            monkeypatch.setattr(state.execution.executions, "get", current)
            release.set()
            result = await query
            assert result.output == {"answer": "canonical"}
        assert callbacks == []
        # A later legitimate terminal handoff is not blocked by the completed consumer.
        await service.request_terminal_handoff("execution", tenant_id="tenant")
        assert callbacks == [("execution", "tenant")]
    finally:
        release.set()
        await state.close()


@pytest.mark.parametrize("method", ("result", "wait", "result_payload_size"))
@pytest.mark.parametrize("change", ("missing", "nonterminal", "cancel"))
async def test_custom_result_wait_preserves_fresh_handoff_and_consumer_cleanup(
    monkeypatch: pytest.MonkeyPatch, change: str, method: str,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    entered, release = asyncio.Event(), asyncio.Event()
    callbacks = []
    try:
        record = await _seed(state)
        repository = state.execution.executions

        class CustomResult(ExecutionRepositoryImpl):
            async def get_result(self, execution_id, *, tenant_id):
                entered.set()
                await release.wait()
                return record.result

        selected = CustomResult(repository.state_store, namespace="result-read", tenant_id="tenant")

        async def terminal(execution_id, *, tenant_id):
            callbacks.append((execution_id, tenant_id))
            return True

        service = _service(state, repository=selected, release=terminal)
        query = asyncio.create_task(getattr(service, method)("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), 2)
        if change == "cancel":
            query.cancel()
            with pytest.raises(asyncio.CancelledError):
                await query
        else:
            async def current(*args, **kwargs):
                return None if change == "missing" else replace(record, status=ExecutionStatus.STARTED, result=None)

            monkeypatch.setattr(selected, "get", current)
            release.set()
            value = await query
            if method == "result_payload_size":
                assert value == record.result.output.size
            else:
                assert value.output == {"text": "result"}
        assert callbacks == []
        await service.request_terminal_handoff("execution", tenant_id="tenant")
        assert callbacks == [("execution", "tenant")]
    finally:
        release.set()
        await state.close()


@pytest.mark.parametrize("history,method", ((False, "result"), (False, "wait"), (True, "result")))
async def test_denied_result_read_does_not_resolve_payload(history: bool, method: str) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        await _seed(state)

        class Authorization:
            async def authorize(self, principal, action, resource):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        with pytest.raises(AIError) as caught:
            await getattr(_service(state, authorization=Authorization(), history=history), method)("execution", principal=_PRINCIPAL)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("status,missing_result", (
    (ExecutionStatus.SUCCEEDED, True),
    (ExecutionStatus.FAILED, False),
    (ExecutionStatus.CANCELLED, False),
    (ExecutionStatus.STARTED, False),
))
async def test_history_result_uses_fresh_terminal_state_after_authorization(
    status: ExecutionStatus, missing_result: bool,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        code = (ErrorCode.EXECUTION_CANCELLED if status is ExecutionStatus.CANCELLED
                else ErrorCode.INTERNAL_ERROR)
        current = record

        class Authorization:
            async def authorize(self, principal, action, resource):
                nonlocal current
                await asyncio.sleep(0)
                current = replace(
                    record, status=status,
                    result=None if missing_result else replace(record.result, output=None),
                    error_code=code.value if status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED} else None,
                )
                await state.execution.executions.compare_and_swap(
                    "execution", tenant_id="tenant", expected_revision=record.revision,
                    next_record=replace(current, revision=record.revision + 1),
                )
        service = _service(state, authorization=Authorization(), history=True)
        if status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await service.result("execution", principal=_PRINCIPAL)
            assert result.status is status and result.output is None
            assert result.error_code == code.value
        else:
            with pytest.raises(AIError) as caught:
                await service.result("execution", principal=_PRINCIPAL)
            assert caught.value.code is (ErrorCode.STORAGE_INTEGRITY_ERROR if missing_result
                                         else ErrorCode.EXECUTION_NOT_READY)
    finally:
        await state.close()


@pytest.mark.parametrize("method", ("result", "result_payload_size", "wait"))
@pytest.mark.parametrize("policy_kind", ("independent", "subclass"))
async def test_result_reauthorizes_owner_changed_during_custom_authorization(method, policy_kind):
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="result-read", tenant_id="tenant")
    try:
        record = await _seed(state)
        owners = []
        policy = TenantAuthorizationPolicy()

        class Authorization(TenantAuthorizationPolicy if policy_kind == "subclass" else object):
            async def authorize(self, principal, action, resource):
                owners.append(resource.owner_principal_id)
                await policy.authorize(principal, action, resource)
                await asyncio.sleep(0)
                await state.execution.executions.compare_and_swap(
                    "execution", tenant_id="tenant", expected_revision=record.revision,
                    next_record=replace(record, revision=record.revision + 1, principal_id="other"),
                )

        with pytest.raises(AIError) as raised:
            await getattr(_service(state, authorization=Authorization()), method)("execution", principal=_PRINCIPAL)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert owners == ["principal", "other"]
    finally:
        await state.close()


async def test_result_projection_skips_archived_input_and_preserves_full_execution(tmp_path: Path) -> None:
    from linktools.ai.model import StoredUserInput

    async with Persistence.open("result-projection", storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        original = _execution()
        original = replace(original, status=ExecutionStatus.SUCCEEDED,
            stored_user_input=StoredUserInput("text", StoredPayload.inline_text("input" * 20000)),
            result=_result(original.execution_id, "inline", original.created_at))
        await state.execution.executions.create_execution(original)
        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
        statements = []

        def capture(_connection, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            result = await _service(state).result(original.execution_id, principal=_PRINCIPAL)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert result.output == {"text": "result"}
        assert len(statements) == 1
        loaded = await state.execution.executions.get(original.execution_id, tenant_id="tenant")
        assert loaded.binding == original.binding
        assert loaded.stored_user_input == original.stored_user_input


@pytest.mark.parametrize("corruption", ("principal", "diagnostics", "completion"))
async def test_result_projection_rejects_invalid_owner_fields(corruption: str) -> None:
    from copy import deepcopy
    from linktools.ai.errors import ErrorDiagnostics
    from linktools.ai.persistence.codec import encode_domain

    async with Persistence.open("result-integrity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        original = await _seed(state)
        repository = state.execution.executions

        async def corrupt(transaction):
            key = repository.execution_record_key(original.execution_id)
            record = await transaction.get_record(key)
            data = deepcopy(dict(record.data))
            fields = data["value"]["payload"]["fields"]
            if corruption == "principal":
                fields["principal_id"] = ""
            elif corruption == "diagnostics":
                fields["error_diagnostics"] = encode_domain(ErrorDiagnostics("Error", "failed", "a" * 64))
            else:
                fields["completed_at"] = encode_domain(original.updated_at)
                fields["status"] = encode_domain(ExecutionStatus.STARTED)
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, data=data))

        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as invalid:
            await _service(state).result_payload_size(original.execution_id, principal=_PRINCIPAL)
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
