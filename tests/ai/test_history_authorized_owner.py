#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History checks the actual owner after authorization without a duplicate header."""

from pathlib import Path

import pytest
from sqlalchemy import event, text

from linktools.ai.core import ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history._projection import StepExecutionHistoryReader
from linktools.ai.history._service import DefaultExecutionHistoryService
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionRecord

from .test_history_projection_conformance import _record


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "owner", "source"))
async def test_history_checks_post_authorization_owner_without_masking_source_errors(
    tmp_path: Path, damage: str,
) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "history.db")
    async with Persistence.open("history", tenant_id="tenant", storage=config) as state:
        await state.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 0))
    counts = []
    for optimized in (False, True):
        async with Persistence.open("history", tenant_id="tenant", storage=config) as state:
            repository = state.execution.executions
            reader = StepExecutionHistoryReader(namespace="history", executions=repository,
                store=state.run_store.read_store(PersistenceDomain.TRACE),
                cursor_signer=HmacCursorSigner("history", b"history-key"))

            class Authorization(TenantAuthorizationPolicy):
                async def authorize(self, principal: object, action: object, resource: object) -> None:
                    await super().authorize(principal, action, resource)
                    if damage == "owner":
                        await repository.state_store.mutate(lambda tx: tx.delete_record(
                            repository.execution_record_key("execution")))
                    elif damage == "source":
                        async with repository.state_store.storage_group.context.engine.begin() as connection:
                            await connection.execute(text("DELETE FROM ai_state_records WHERE kind = 'archive_content'"))

            class CustomService(DefaultExecutionHistoryService):
                pass

            service = (DefaultExecutionHistoryService if optimized else CustomService)(
                repository, Authorization(), reader)
            statements: list[str] = []

            def observe(connection: object, cursor: object, statement: str, parameters: object,
                        context: object, executemany: bool) -> None:
                if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                    statements.append(statement)

            engine = repository.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", observe)
            try:
                if damage == "none":
                    page = await service.history("execution", principal=Principal("principal", "tenant"))
                    assert page.items == ()
                    counts.append(len(statements))
                else:
                    with pytest.raises(AIError) as caught:
                        await service.history("execution", principal=Principal("principal", "tenant"))
                    expected = ErrorCode.AUTHORIZATION_DENIED if damage == "owner" else ErrorCode.STORAGE_INTEGRITY_ERROR
                    assert caught.value.code is expected
            finally:
                event.remove(engine, "before_cursor_execute", observe)
        if damage != "none":
            # Each authorization callback damages a fresh durable owner.
            async with Persistence.open("history", tenant_id="tenant", storage=config) as state:
                repository = state.execution.executions
                await repository.state_store.mutate(lambda tx: tx.delete_record(repository.execution_record_key("execution")))
                await repository.create_execution(_record(ExecutionStatus.STARTED, 0))
    if damage == "none":
        assert counts[1] == counts[0] - 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", (False, True))
async def test_history_preserves_pending_checks(tmp_path: Path, pending: bool) -> None:
    async with Persistence.open("history", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "pending.db")) as state:
        repository = state.execution.executions
        record = _record(ExecutionStatus.STARTED, 0)
        if not pending:
            await repository.create_execution(record)
        calls: list[str] = []

        async def service_pending(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
            calls.append("service")
            return record if pending else None

        async def reader_pending(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
            calls.append("reader")
            return record if pending else None

        reader = StepExecutionHistoryReader(namespace="history", executions=repository,
            store=state.run_store.read_store(PersistenceDomain.TRACE),
            cursor_signer=HmacCursorSigner("history", b"history-key"))
        reader.bind_pending_execution_reader(reader_pending)
        service = DefaultExecutionHistoryService(repository, TenantAuthorizationPolicy(), reader,
            pending_execution=service_pending)
        assert (await service.history("execution", principal=Principal("principal", "tenant"))).items == ()
        assert calls == ["service", "service", "reader"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("custom_reader", "different_repository"))
@pytest.mark.parametrize("deleted", (False, True))
async def test_history_keeps_custom_and_different_repository_header_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, deleted: bool,
) -> None:
    async with Persistence.open("history", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "owner.db")) as state:
        async with Persistence.open("history", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as other:
            repository = state.execution.executions
            await repository.create_execution(_record(ExecutionStatus.STARTED, 0))
            await other.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 0))
            headers = 0
            original = repository.get_header

            async def header(identity: str, *, tenant_id: str) -> object:
                nonlocal headers
                headers += 1
                return await original(identity, tenant_id=tenant_id)

            monkeypatch.setattr(repository, "get_header", header)

            class CustomReader(StepExecutionHistoryReader):
                pass

            reader_state = other if mode == "different_repository" else state
            reader = (CustomReader if mode == "custom_reader" else StepExecutionHistoryReader)(
                namespace="history", executions=reader_state.execution.executions,
                store=reader_state.run_store.read_store(PersistenceDomain.TRACE),
                cursor_signer=HmacCursorSigner("history", b"history-key"))

            class Authorization(TenantAuthorizationPolicy):
                async def authorize(self, principal: object, action: object, resource: object) -> None:
                    await super().authorize(principal, action, resource)
                    if deleted:
                        await repository.state_store.mutate(lambda tx: tx.delete_record(
                            repository.execution_record_key("execution")))

            service = DefaultExecutionHistoryService(repository, Authorization(), reader)
            if deleted:
                with pytest.raises(AIError) as caught:
                    await service.history("execution", principal=Principal("principal", "tenant"))
                assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
            else:
                assert (await service.history("execution", principal=Principal("principal", "tenant"))).items == ()
            assert headers == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_owner", (False, True))
async def test_late_pending_history_does_not_consume_durable_body(
    tmp_path: Path, delete_owner: bool,
) -> None:
    async with Persistence.open("history", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "late.db")) as state:
        repository = state.execution.executions
        record = _record(ExecutionStatus.STARTED, 0)
        await repository.create_execution(record)
        async with repository.state_store.storage_group.context.engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages"))
        calls = 0

        async def pending(execution_id: str, *, tenant_id: str) -> ExecutionRecord | None:
            nonlocal calls
            calls += 1
            if calls < 3:
                return None
            if delete_owner:
                await repository.state_store.mutate(lambda tx: tx.delete_record(
                    repository.execution_record_key("execution")))
            return record

        reader = StepExecutionHistoryReader(namespace="history", executions=repository,
            store=state.run_store.read_store(PersistenceDomain.TRACE),
            cursor_signer=HmacCursorSigner("history", b"history-key"))
        reader.bind_pending_execution_reader(pending)
        service = DefaultExecutionHistoryService(repository, TenantAuthorizationPolicy(), reader,
            pending_execution=pending)
        if delete_owner:
            with pytest.raises(AIError) as caught:
                await service.history("execution", principal=Principal("principal", "tenant"))
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        else:
            assert (await service.history("execution", principal=Principal("principal", "tenant"))).items == ()
        assert calls == 3
