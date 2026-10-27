#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized model metadata captures retain freshness with bounded read I/O."""

from linktools.ai.persistence import Persistence
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
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.history._projection import StepExecutionHistoryReader
from linktools.ai.history._service import DefaultExecutionHistoryService
from linktools.ai.persistence._model_interaction import StagedModelInteraction
from linktools.ai.execution import ModelInteractionReadBoundary, UsageReadCutoff
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence.state import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio
_NAMESPACE = "model-capture"
_PRINCIPAL = Principal("owner", "tenant")


def _reader(state, *, store=None, executions=None, reader_type=StepExecutionHistoryReader):
    return reader_type(
        namespace=_NAMESPACE,
        executions=state.execution.executions if executions is None else executions,
        store=state.run_store.read_store(PersistenceDomain.TRACE) if store is None else store,
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
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
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
        monkeypatch.setattr(state.object_store(PersistenceDomain.TRACE), "open", unexpected_object)
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
    state = Persistence(PersistenceStorage.in_memory())
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
    state = Persistence(PersistenceStorage.in_memory())
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
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "execution.db"))
    other = Persistence(PersistenceStorage.sqlite(tmp_path / "history.db"))
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
        archive = other.run_store.read_store(PersistenceDomain.TRACE)
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
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
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
    state = Persistence(PersistenceStorage.in_memory())
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
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "capture.db"))
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
    state = Persistence(PersistenceStorage.in_memory())
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
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
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
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
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
    state = Persistence(PersistenceStorage.in_memory())
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
@pytest.mark.parametrize("operation", ("capture", "metadata"))
async def test_model_reads_keep_requested_execution_as_lookup_scope(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, custom_repository: bool, operation: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        await repository.create_with_history_head(replace(
            _execution(), execution_id="other", root_execution_id="other",
        ))
        other_run = await _publish(state, execution_id="other")
        state.run_store.stage_model_interaction(StagedModelInteraction(
            other_run.agent_run_id, 1, 1, "agent", None, {}, None, None, None,
            "FAILED", "MODEL_API_ERROR", 1, None, other_run.started_at, other_run.started_at,
        ))
        await state.run_store.flush_execution_projection(other_run.agent_run_id, execution_id="other")
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
        service = _service(state, reader=reader, executions=selected, authorization=Authorization())
        if operation == "metadata":
            values = await _metadata(service) if entrypoint == "service" else await reader.read_model_interaction_metadata(
                "execution", tenant_id="tenant", agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=1,
            )
            assert [(value.execution_id, value.model_request_seq, value.status) for value in values] == [("execution", 1, "RUNNING")]
        else:
            boundary = (await service.capture_model_interaction_cutoffs("execution", principal=_PRINCIPAL)
                        if entrypoint == "service" else await reader.capture_model_interaction_cutoffs("execution", tenant_id="tenant"))
            assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 1),)
    finally:
        await state.close()


async def _metadata(service, *, principal=_PRINCIPAL, agent_run_seq=1, through=1, limit=200):
    return await service.read_model_interaction_metadata(
        "execution", principal=principal, agent_run_seq=agent_run_seq,
        after_model_request_seq=0, through_model_request_seq=through, limit=limit,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_metadata_rows_share_fresh_authorized_scope_without_body_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        get, rows = repository.get, archive.read_model_interactions_with_run
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
        async def tracked_rows(*args, **kwargs):
            scopes.append(active_state_scope())
            return await rows(*args, **kwargs)
        async def unexpected_body(*args, **kwargs):
            raise AssertionError("metadata rows must not resolve request or response bodies")
        def unexpected_object(*args, **kwargs):
            raise AssertionError("metadata rows must not open object bodies")
        monkeypatch.setattr(repository, "get", tracked_get)
        monkeypatch.setattr(archive, "read_model_interactions_with_run", tracked_rows)
        monkeypatch.setattr(archive, "resolve_model_interactions", unexpected_body)
        monkeypatch.setattr(state.object_store(PersistenceDomain.TRACE), "open", unexpected_object)
        values = await _metadata(_service(state, authorization=Authorization()))
        assert [(value.execution_id, value.agent_run_seq, value.model_request_seq, value.status)
                for value in values] == [("execution", 1, 1, "RUNNING")]
        assert all(value.request == {} and value.response is None and not value.content_included for value in values)
        assert scopes and all(scope is scopes[0] and scope is not None for scope in scopes)
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_metadata_sql_reuses_authorized_record_and_one_snapshot(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "metadata.db"))
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
            values = [await _metadata(_service(state)) for _ in range(3)]
        finally:
            event.remove(engine, "before_cursor_execute", capture)
            event.remove(engine, "begin", begin)
        assert all([(item.execution_id, item.model_request_seq) for item in page] == [("execution", 1)] for page in values)
        assert len(begins) <= 6
        assert sum(statement.lstrip().upper().startswith("SELECT") for statement in statements) <= 15
    finally:
        await state.close()


async def test_metadata_observes_authorization_time_admission_and_header_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
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
        pending = asyncio.create_task(_metadata(service, agent_run_seq=2))
        await asyncio.wait_for(entered.wait(), timeout=5)
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, agent_run_seq=2, revision=current.revision + 1),
        )
        await _publish(state, run_sequence=2)
        proceed.set()
        values = await pending
        assert [(value.agent_run_seq, value.model_request_seq) for value in values] == [(2, 1)]
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


@pytest.mark.parametrize("same_repository", (False, True))
async def test_metadata_preserves_custom_repository_reader_authority(same_repository: bool) -> None:
    state = Persistence(PersistenceStorage.in_memory())
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
        values = await _metadata(_service(
            state, executions=authorizing, reader=_reader(state, executions=reading),
        ), agent_run_seq=2)
        assert [(value.agent_run_seq, value.model_request_seq) for value in values] == [(2, 1)]
    finally:
        await state.close()


@pytest.mark.parametrize("reader_kind", ("protocol", "subclass"))
async def test_metadata_preserves_injected_reader_protocol_and_overrides(reader_kind: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    expected, calls = (object(),), []
    try:
        await _seed(state)
        class Reader(StepExecutionHistoryReader if reader_kind == "subclass" else object):
            async def read_model_interaction_metadata(
                self, execution_id, *, tenant_id, agent_run_seq,
                after_model_request_seq, through_model_request_seq, limit,
            ):
                assert active_state_scope() is None
                calls.append((execution_id, tenant_id, agent_run_seq, after_model_request_seq, through_model_request_seq, limit))
                return expected
        reader = _reader(state, reader_type=Reader) if reader_kind == "subclass" else Reader()
        assert await _metadata(_service(state, reader=reader)) is expected
        assert calls == [("execution", "tenant", 1, 0, 1, 200)]
    finally:
        await state.close()


@pytest.mark.parametrize("owner", ("split", "different_repository", "custom_archive"))
async def test_metadata_preserves_independent_owner_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str,
) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "execution.db"))
    other = Persistence(PersistenceStorage.sqlite(tmp_path / "history.db"))
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
            raise AssertionError("read-side header must not replace authorization authority")
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                assert resource.owner_principal_id == "authorizing-owner"
        monkeypatch.setattr(state.execution.executions, "get_header", authorizing_header)
        monkeypatch.setattr(other.execution.executions, "get_header", unexpected_header)
        archive = other.run_store.read_store(PersistenceDomain.TRACE)
        get_run, rows = archive.get_agent_run, archive.list_model_interactions
        combined = archive.read_model_interactions_with_run
        async def checked_run(*args, **kwargs):
            assert active_state_scope() is None
            return await get_run(*args, **kwargs)
        async def checked_rows(*args, **kwargs):
            assert active_state_scope() is None
            return await rows(*args, **kwargs)
        async def checked_combined(*args, **kwargs):
            assert active_state_scope() is None
            return await combined(*args, **kwargs)
        class Archive:
            get_agent_run = staticmethod(checked_run)
            list_model_interactions = staticmethod(checked_rows)
        monkeypatch.setattr(archive, "read_model_interactions_with_run", checked_combined)
        monkeypatch.setattr(archive, "get_agent_run", checked_run)
        monkeypatch.setattr(archive, "list_model_interactions", checked_rows)
        reader = _reader(
            state, store=Archive() if owner == "custom_archive" else archive,
            executions=other.execution.executions if owner == "different_repository" else state.execution.executions,
        )
        values = await _metadata(_service(state, reader=reader, authorization=Authorization()), through=2)
        assert [(value.execution_id, value.model_request_seq) for value in values] == [("execution", 1), ("execution", 2)]
        assert active_state_scope() is None
    finally:
        await other.close()
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_metadata_cancellation_releases_shared_read_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        rows = archive.read_model_interactions_with_run
        entered, blocked = asyncio.Event(), asyncio.Event()
        async def blocked_rows(*args, **kwargs):
            assert active_state_scope() is not None
            entered.set()
            await blocked.wait()
            return await rows(*args, **kwargs)
        monkeypatch.setattr(archive, "read_model_interactions_with_run", blocked_rows)
        service = _service(state)
        pending = asyncio.create_task(_metadata(service))
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert active_state_scope() is None
        monkeypatch.setattr(archive, "read_model_interactions_with_run", rows)
        values = await _metadata(service)
        assert [value.model_request_seq for value in values] == [1]
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await state.close()


@pytest.mark.parametrize("entrypoint,phase,limit,through,expected", (
    ("service", "denied", 0, 1, ErrorCode.AUTHORIZATION_DENIED),
    ("service", "tenant", 0, 1, ErrorCode.AUTHORIZATION_DENIED),
    ("service", "postauth_missing", 0, 1, ErrorCode.AUTHORIZATION_DENIED),
    ("service", "postauth_missing", 1, 0, ErrorCode.AUTHORIZATION_DENIED),
    ("service", "reader_missing", 0, 1, ErrorCode.PAGE_LIMIT_INVALID),
    ("service", "reader_missing", 1, 1, ErrorCode.STORAGE_NOT_FOUND),
    ("service", "ready", 0, 1, ErrorCode.PAGE_LIMIT_INVALID),
    ("service", "ready", 1, -1, ErrorCode.REQUEST_FIELD_INVALID),
    ("reader", "reader_missing", 0, 1, ErrorCode.PAGE_LIMIT_INVALID),
    ("reader", "reader_missing", 1, -1, ErrorCode.REQUEST_FIELD_INVALID),
    ("reader", "reader_missing", 1, 1, ErrorCode.STORAGE_NOT_FOUND),
))
async def test_metadata_preserves_authority_validation_and_missing_record_order(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, phase: str, limit: int, through: int,
    expected: ErrorCode,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        get = repository.get
        authorized = False
        class Authorization:
            async def authorize(self, principal, action, resource):
                nonlocal authorized
                if phase == "denied":
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                authorized = True
        async def get_after_auth(*args, **kwargs):
            return None if authorized and phase == "postauth_missing" else await get(*args, **kwargs)
        class MissingReaderRepository:
            async def get(self, execution_id, *, tenant_id):
                return None
        async def unexpected(*args, **kwargs):
            raise AssertionError("model rows must not be read before authority and request validation")
        monkeypatch.setattr(repository, "get", get_after_auth)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        monkeypatch.setattr(archive, "get_agent_run", unexpected)
        monkeypatch.setattr(archive, "read_model_interactions_with_run", unexpected)
        reader = _reader(state, executions=MissingReaderRepository() if phase == "reader_missing" else repository)
        with pytest.raises(AIError) as raised:
            if entrypoint == "service":
                principal = Principal("owner", "other") if phase == "tenant" else _PRINCIPAL
                await _metadata(_service(state, reader=reader, authorization=Authorization()), principal=principal, limit=limit, through=through)
            else:
                await reader.read_model_interaction_metadata(
                    "execution", tenant_id="tenant", agent_run_seq=1,
                    after_model_request_seq=0, through_model_request_seq=through, limit=limit,
                )
        assert raised.value.code is expected
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_metadata_batches_run_and_interaction_records_without_skipping_high_water(tmp_path: Path) -> None:
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "metadata-batch.db"))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        run = await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        engine = archive.state_store.storage_group.context.engine.sync_engine
        statements = []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.lstrip().upper())
        event.listen(engine, "before_cursor_execute", capture)
        try:
            async def separate(transaction):
                del transaction
                return (await archive.get_agent_run(agent_run_id=run.agent_run_id),
                        await archive.list_model_interactions(agent_run_id=run.agent_run_id, limit=1))
            separate_page = await archive.state_store.read(separate)
            separate_selects = sum(sql.startswith("SELECT") for sql in statements)
            statements.clear()
            combined_page = await archive.read_model_interactions_with_run(agent_run_id=run.agent_run_id, limit=1)
            combined_selects = sum(sql.startswith("SELECT") for sql in statements)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert combined_page == separate_page
        empty_run, empty_page = await archive.read_model_interactions_with_run(
            agent_run_id=run.agent_run_id, after_model_request_seq=1, limit=1,
        )
        assert empty_run == run and empty_page == []
        assert combined_selects < separate_selects
        print("run/page SELECT separate, batched:", separate_selects, combined_selects)
        # A row beyond the durable sequence boundary must not become visible.
        await archive.state_store.mutate(lambda transaction: transaction.delete_sequence(archive._sequence(run.agent_run_id, "interaction")))
        with pytest.raises(AIError) as raised:
            await _metadata(_service(state))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("missing", ("run", "interaction"))
async def test_metadata_batch_rejects_missing_durable_rows(tmp_path: Path, backend: str, missing: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        run = await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        key = archive._agent_run_key(run.agent_run_id) if missing == "run" else archive._interaction_key(run.agent_run_id, 1)
        await archive.state_store.mutate(lambda transaction: transaction.delete_record(key))
        with pytest.raises(AIError) as raised:
            await _metadata(_service(state))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_metadata_batch_rejects_fact_in_place_of_authoritative_record(tmp_path: Path) -> None:
    from .test_model_interaction_lifecycle_paging import _terminal
    from linktools.ai.persistence._step_archive import _encode_step
    from linktools.ai.persistence.state import StoredFact
    from linktools.ai.core import canonical_sha256

    state = Persistence(PersistenceStorage.sqlite(tmp_path / "missing-record-metadata.db"))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        run = await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        value, = await archive.list_model_interactions(agent_run_id=run.agent_run_id)
        terminal = _terminal(value)
        fact = StoredFact(
            archive._stream(run.agent_run_id, "interaction"), 1,
            archive._agent_run_key(run.agent_run_id), "model_interaction",
            bytes.fromhex(canonical_sha256({"agent_run_id": terminal.agent_run_id, "model_request_seq": 1})), terminal.status, _encode_step(terminal),
        )
        async def seed(transaction):
            owner = await transaction.get_record(fact.owner_key_digest)
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.delete_record(archive._interaction_key(run.agent_run_id, 1))
            await transaction.insert_fact(fact)
        await archive.state_store.mutate(seed)
        with pytest.raises(AIError) as raised:
            await _metadata(_service(state))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


async def test_metadata_batch_preserves_custom_archive_overrides() -> None:
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        calls = []
        class CustomArchive(ModelInteractionStateStepArchive):
            async def get_agent_run(self, *, agent_run_id):
                calls.append("run")
                return await super().get_agent_run(agent_run_id=agent_run_id)
            async def list_model_interactions(self, **kwargs):
                calls.append("interactions")
                return await super().list_model_interactions(**kwargs)
            async def read_model_interactions_with_run(self, **kwargs):
                raise AssertionError("custom archive must retain its independent operations")
        custom = CustomArchive(
            archive.state_store, object_store=None, namespace=_NAMESPACE,
            tenant_id="tenant", runtime_domain=PersistenceDomain.TRACE,
        )
        page = await _metadata(_service(state, reader=_reader(state, store=custom)))
        assert [item.model_request_seq for item in page] == [1]
        assert calls == ["run", "interactions"]
    finally:
        await state.close()
