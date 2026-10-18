#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subscriptions preserve fresh authorization without duplicate owner queries."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import HmacCursorSigner
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_model_capture_reads import _NAMESPACE, _PRINCIPAL, _service
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


def _reader(state, *, executions=None, reader_type=StepExecutionHistoryReader):
    return reader_type(
        namespace=_NAMESPACE,
        executions=state.execution.executions if executions is None else executions,
        store=state.run_store.read_store(RuntimeDomain.TRACE),
        cursor_signer=HmacCursorSigner("subscription", b"subscription-key"),
        notifications=state.run_store,
    )


async def test_model_subscription_eliminates_duplicate_sql_owner_lookup(tmp_path: Path) -> None:
    state = _storage("sqlite", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine

        class CustomReader(StepExecutionHistoryReader):
            pass

        counts = []
        for reader_type in (CustomReader, StepExecutionHistoryReader):
            statements = []

            def capture(connection, cursor, statement, parameters, context, many):
                statements.append(statement.lstrip().upper())

            event.listen(engine, "before_cursor_execute", capture)
            try:
                subscription = await _service(state, reader=_reader(state, reader_type=reader_type)).subscribe_model_interactions(
                    "execution", principal=_PRINCIPAL,
                )
                assert subscription is not None and subscription.generation == 0
                await subscription.close()
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            counts.append((len(statements), sum(sql.startswith("SELECT") for sql in statements)))
        assert counts[1][0] < counts[0][0]
        assert counts[1][1] < counts[0][1]
        print("subscription SQL/SELECT baseline, grouped:", counts)
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("denial", ("policy", "missing_after_authorization", "tenant"))
async def test_model_subscription_fails_closed_before_registering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, denial: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        principal = replace(_PRINCIPAL, tenant_id="other") if denial == "tenant" else _PRINCIPAL

        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                if denial == "policy":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

                async def missing(*args, **kwargs):
                    return None
                monkeypatch.setattr(state.execution.executions, "get", missing)

        def forbidden(*args, **kwargs):
            raise AssertionError("Unauthorized subscription must not be registered")

        monkeypatch.setattr(state.run_store, "subscribe_model_interactions", forbidden)
        with pytest.raises(AIError) as caught:
            await _service(state, reader=_reader(state), authorization=Authorization()).subscribe_model_interactions(
                "execution", principal=principal,
            )
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("owner", ("subclass", "different"))
async def test_model_subscription_preserves_custom_and_different_reader_authority(
    tmp_path: Path, owner: str,
) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        repository = state.execution.executions
        calls = []

        class Custom(ExecutionRepositoryImpl):
            async def get(self, execution_id: str, *, tenant_id: str):
                calls.append(execution_id)
                # Header and post-authorization reads succeed; the reader still
                # owns an independent existence decision for custom repositories.
                if len(calls) >= 3:
                    return None
                return await super().get(execution_id, tenant_id=tenant_id)

        if owner == "subclass":
            selected = Custom(repository.state_store, namespace=_NAMESPACE, tenant_id="tenant")
            reader = _reader(state, executions=selected)
        else:
            selected = repository
            different = ExecutionRepositoryImpl(repository.state_store, namespace="different", tenant_id="tenant")
            reader = _reader(state, executions=different)
        with pytest.raises(AIError) as caught:
            await _service(state, reader=reader, executions=selected).subscribe_model_interactions(
                "execution", principal=_PRINCIPAL,
            )
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
        if owner == "subclass":
            assert len(calls) == 3
    finally:
        await state.close()


async def test_model_subscription_preserves_custom_reader_override(tmp_path: Path) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(_execution())
        calls = []

        class CustomReader(StepExecutionHistoryReader):
            async def subscribe_model_interactions(self, execution_id: str, *, tenant_id: str):
                calls.append((execution_id, tenant_id))
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

        with pytest.raises(AIError) as caught:
            await _service(state, reader=_reader(state, reader_type=CustomReader)).subscribe_model_interactions(
                "execution", principal=_PRINCIPAL,
            )
        assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert calls == [("execution", "tenant")]
    finally:
        await state.close()
