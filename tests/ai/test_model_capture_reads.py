#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized model metadata captures retain freshness with bounded read I/O."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import (
    ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy,
    agent_conversation_id, agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._model_interaction import StagedModelInteraction
from linktools.ai.runtime.service_api import ModelInteractionReadBoundary, UsageReadCutoff
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio
_NAMESPACE = "model-capture"
_PRINCIPAL = Principal("owner", "tenant")


def _reader(state, *, store=None, executions=None, reader_type=StepExecutionHistoryReader):
    return reader_type(
        namespace=_NAMESPACE,
        executions=state.execution.executions if executions is None else executions,
        store=state.run_store.read_store(RuntimeDomain.EXECUTION) if store is None else store,
        cursor_signer=HmacCursorSigner("capture", b"capture-key"),
    )


def _service(state, *, reader=None, authorization=None, executions=None):
    return DefaultExecutionHistoryService(
        state.execution.executions if executions is None else executions,
        TenantAuthorizationPolicy() if authorization is None else authorization,
        _reader(state) if reader is None else reader,
    )


async def _publish(state, *, execution_id="execution", run_sequence=1, request_sequence=1):
    run = AgentRunRecord(
        agent_run_id(
            namespace=_NAMESPACE, tenant_id="tenant", execution_id=execution_id,
            agent_run_seq=run_sequence,
        ),
        agent_conversation_id=agent_conversation_id(
            namespace=_NAMESPACE, tenant_id="tenant", execution_id=execution_id,
        ),
        agent_id="agent", metadata={"agent_run_seq": str(run_sequence)},
    )
    if request_sequence == 1:
        await state.run_store.register_agent_run(run)
    state.run_store.stage_model_interaction(StagedModelInteraction(
        run.agent_run_id, request_sequence, request_sequence, "agent", None,
        {}, None, None, None, "RUNNING", None, None, None, run.started_at, None,
    ))
    await state.run_store.flush_execution_projection(run.agent_run_id, execution_id=execution_id)
    return run


async def _seed(state):
    await state.execution.executions.create_with_history_head(_execution())
    return await _publish(state)


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_capture_reads_fresh_metadata_in_one_post_authorization_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        get, capture = repository.get, archive.capture_history
        scopes = []
        authorized = False

        class Authorization:
            async def authorize(self, principal, action, resource):
                nonlocal authorized
                assert active_state_scope() is None
                authorized = True

        async def tracked_get(*args, **kwargs):
            if authorized:
                scopes.append(active_state_scope())
            return await get(*args, **kwargs)

        async def tracked_capture(*args, **kwargs):
            scopes.append(active_state_scope())
            return await capture(*args, **kwargs)

        async def unexpected_body(*args, **kwargs):
            raise AssertionError("cutoff capture must not resolve model or transcript bodies")

        def unexpected_object(*args, **kwargs):
            raise AssertionError("cutoff capture must not read object bodies")

        monkeypatch.setattr(repository, "get", tracked_get)
        monkeypatch.setattr(archive, "capture_history", tracked_capture)
        monkeypatch.setattr(archive, "resolve_model_interactions", unexpected_body)
        monkeypatch.setattr(archive, "read_pending_message", unexpected_body)
        monkeypatch.setattr(state.object_store(RuntimeDomain.EXECUTION), "open", unexpected_object)
        boundary = await _service(state, authorization=Authorization()).capture_model_interaction_cutoffs(
            "execution", principal=_PRINCIPAL,
        )
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),)
        assert scopes and all(scope is scopes[0] and scope is not None for scope in scopes)
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_capture_preserves_header_authority_and_observes_post_authorization_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        repository = state.execution.executions
        header = repository.get_header
        entered, proceed = asyncio.Event(), asyncio.Event()

        async def custom_header(*args, **kwargs):
            return replace(await header(*args, **kwargs), owner_principal_id="custom-owner")

        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                assert resource.owner_principal_id == "custom-owner"
                entered.set()
                await proceed.wait()

        monkeypatch.setattr(repository, "get_header", custom_header)
        service = _service(state, authorization=Authorization())
        pending = asyncio.create_task(service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), timeout=5)
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, agent_run_seq=2, revision=current.revision + 1),
        )
        await _publish(state, run_sequence=2)
        proceed.set()
        boundary = await pending
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1), UsageReadCutoff("execution", 2, 1))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


@pytest.mark.parametrize("reader_kind", ("protocol", "subclass"))
async def test_capture_preserves_injected_reader_protocol_and_overrides(reader_kind: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    expected = ModelInteractionReadBoundary((), (), False, False)
    calls = []
    try:
        await _seed(state)
        class Reader(StepExecutionHistoryReader if reader_kind == "subclass" else object):
            async def capture_model_interaction_cutoffs(self, execution_id, *, tenant_id):
                assert active_state_scope() is None
                calls.append((execution_id, tenant_id))
                return expected
        reader = _reader(state, reader_type=Reader) if reader_kind == "subclass" else Reader()
        actual = await _service(state, reader=reader).capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert actual is expected
        assert calls == [("execution", "tenant")]
    finally:
        await state.close()


@pytest.mark.parametrize("mismatched_repository", (False, True))
async def test_capture_keeps_independent_storage_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatched_repository: bool,
) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "execution.db")
    other = RuntimeStorage.sqlite(tmp_path / "history.db")
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    await other.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        await _seed(other)
        await _publish(other, request_sequence=2)
        header = state.execution.executions.get_header
        async def authorizing_header(*args, **kwargs):
            return replace(await header(*args, **kwargs), owner_principal_id="authorizing-owner")
        async def unexpected_header(*args, **kwargs):
            raise AssertionError("the reading repository must not replace service authority")
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                assert resource.owner_principal_id == "authorizing-owner"
        monkeypatch.setattr(state.execution.executions, "get_header", authorizing_header)
        monkeypatch.setattr(other.execution.executions, "get_header", unexpected_header)
        archive = other.run_store.read_store(RuntimeDomain.EXECUTION)
        capture = archive.capture_history
        async def checked_capture(*args, **kwargs):
            assert active_state_scope() is None
            return await capture(*args, **kwargs)
        monkeypatch.setattr(archive, "capture_history", checked_capture)
        reader = _reader(
            state, store=archive,
            executions=other.execution.executions if mismatched_repository else state.execution.executions,
        )
        boundary = await _service(state, reader=reader, authorization=Authorization()).capture_model_interaction_cutoffs(
            "execution", principal=_PRINCIPAL,
        )
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 2),)
        assert active_state_scope() is None
    finally:
        await other.close()
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_capture_releases_metadata_scope_after_observer_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        capture = archive.capture_history
        entered, blocked = asyncio.Event(), asyncio.Event()
        async def blocked_capture(*args, **kwargs):
            assert active_state_scope() is not None
            entered.set()
            await blocked.wait()
            return await capture(*args, **kwargs)
        monkeypatch.setattr(archive, "capture_history", blocked_capture)
        service = _service(state)
        pending = asyncio.create_task(service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert active_state_scope() is None
        monkeypatch.setattr(archive, "capture_history", capture)
        boundary = await service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),)
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


async def test_terminal_capture_observes_late_flush_and_new_run() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, status=ExecutionStatus.FAILED, revision=current.revision + 1),
        )
        service = _service(state)
        first = await service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        await _publish(state, request_sequence=2)
        second = await service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert first.cutoffs == (UsageReadCutoff("execution", 1, 1),)
        assert second.cutoffs == (UsageReadCutoff("execution", 1, 2),)
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, status=ExecutionStatus.STARTED, agent_run_seq=2, revision=current.revision + 1),
        )
        await _publish(state, run_sequence=2)
        third = await service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert third.cutoffs == (UsageReadCutoff("execution", 1, 2), UsageReadCutoff("execution", 2, 1))
    finally:
        await state.close()


async def test_capture_idle_sql_has_one_authority_read_and_one_metadata_snapshot(tmp_path: Path) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "capture.db")
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
        statements, begins = [], []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        def begin(connection):
            begins.append(connection)
        event.listen(engine, "before_cursor_execute", capture)
        event.listen(engine, "begin", begin)
        try:
            boundaries = [
                await _service(state).capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
                for _ in range(3)
            ]
        finally:
            event.remove(engine, "before_cursor_execute", capture)
            event.remove(engine, "begin", begin)
        assert all(boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),) for boundary in boundaries)
        assert len(begins) <= 6
        assert sum(statement.lstrip().upper().startswith("SELECT") for statement in statements) <= 12
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("denied", "disappeared", "tenant"))
@pytest.mark.parametrize("custom_reader", (False, True))
async def test_capture_rejects_failed_authority_before_reading_model_metadata(
    monkeypatch: pytest.MonkeyPatch, failure: str, custom_reader: bool,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        get = repository.get
        authorized = False
        class Authorization:
            async def authorize(self, principal, action, resource):
                nonlocal authorized
                if failure == "denied":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                authorized = True
        async def get_after_auth(*args, **kwargs):
            return None if authorized else await get(*args, **kwargs)
        async def unexpected(*args, **kwargs):
            raise AssertionError("model metadata must not be read after failed authority")
        class Reader:
            capture_model_interaction_cutoffs = unexpected
        monkeypatch.setattr(repository, "get", get_after_auth)
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        monkeypatch.setattr(archive, "capture_history", unexpected)
        service = _service(state, reader=Reader() if custom_reader else None, authorization=Authorization())
        principal = Principal("owner", "other") if failure == "tenant" else _PRINCIPAL
        with pytest.raises(AIError) as raised:
            await service.capture_model_interaction_cutoffs("execution", principal=principal)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert active_state_scope() is None
    finally:
        await state.close()


@pytest.mark.parametrize("custom_owner", ("repository", "archive"))
async def test_capture_does_not_scope_custom_owner_io(custom_owner: str) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        archive = state.run_store.read_store(RuntimeDomain.EXECUTION)
        class Repository:
            async def get_header(self, execution_id, *, tenant_id):
                assert active_state_scope() is None
                return await repository.get_header(execution_id, tenant_id=tenant_id)
            async def get(self, execution_id, *, tenant_id):
                assert active_state_scope() is None
                return await repository.get(execution_id, tenant_id=tenant_id)
        class Archive:
            async def capture_history(self, run_ids):
                assert active_state_scope() is None
                return await archive.capture_history(run_ids)
        selected_repository = Repository() if custom_owner == "repository" else repository
        reader = _reader(
            state, executions=selected_repository,
            store=Archive() if custom_owner == "archive" else archive,
        )
        boundary = await _service(
            state, reader=reader, executions=selected_repository,
        ).capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),)
    finally:
        await state.close()


@pytest.mark.parametrize("same_repository", (False, True))
async def test_capture_preserves_custom_repository_reader_authority(same_repository: bool) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, agent_run_seq=2, revision=current.revision + 1),
        )
        await _publish(state, run_sequence=2)
        class Repository:
            def __init__(self, authorizing):
                self.authorizing = authorizing
            async def get_header(self, execution_id, *, tenant_id):
                return await repository.get_header(execution_id, tenant_id=tenant_id)
            async def get(self, execution_id, *, tenant_id):
                assert active_state_scope() is None
                record = await repository.get(execution_id, tenant_id=tenant_id)
                if self.authorizing:
                    self.authorizing = False
                    return replace(record, agent_run_seq=1)
                return record
        authorizing = Repository(True)
        reading = authorizing if same_repository else Repository(False)
        boundary = await _service(
            state, executions=authorizing, reader=_reader(state, executions=reading),
        ).capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1), UsageReadCutoff("execution", 2, 1))
    finally:
        await state.close()


@pytest.mark.parametrize("entrypoint", ("service", "reader"))
@pytest.mark.parametrize("custom_repository", (False, True))
async def test_capture_keeps_requested_execution_as_lookup_scope(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, custom_repository: bool,
) -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        await repository.create_with_history_head(replace(
            _execution(), execution_id="other", root_execution_id="other",
        ))
        await _publish(state, execution_id="other")
        await _publish(state, execution_id="other", request_sequence=2)
        get = repository.get
        async def alias_shaped_record(execution_id, *, tenant_id):
            record = await get(execution_id, tenant_id=tenant_id)
            return None if record is None else replace(record, execution_id="other")
        class Repository:
            async def get_header(self, execution_id, *, tenant_id):
                return await repository.get_header(execution_id, tenant_id=tenant_id)
            async def get(self, execution_id, *, tenant_id):
                return await alias_shaped_record(execution_id, tenant_id=tenant_id)
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert resource.id == "execution"
        selected = Repository() if custom_repository else repository
        if not custom_repository:
            monkeypatch.setattr(repository, "get", alias_shaped_record)
        reader = _reader(state, executions=selected)
        if entrypoint == "service":
            boundary = await _service(
                state, reader=reader, executions=selected, authorization=Authorization(),
            ).capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
        else:
            boundary = await reader.capture_model_interaction_cutoffs("execution", tenant_id="tenant")
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),)
    finally:
        await state.close()
