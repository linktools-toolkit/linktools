#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded model update reads preserve authority and existing history extensions."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._graph_projection import _GraphModelProjection
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._runtime_history import RuntimeHistory
from linktools.ai.runtime.service_api import ModelInteractionReadBoundary, UsageReadCutoff, project_execution_view
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_model_capture_reads import _NAMESPACE, _PRINCIPAL, _publish, _reader, _seed, _service
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_updates_read_bounded_fresh_suffix_without_content(tmp_path: Path, backend: str) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        await _publish(state, request_sequence=2)
        calls = []
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                calls.append((principal, resource.id))
        history = RuntimeHistory(_service(state, authorization=Authorization()), tenant_id="tenant")
        statements = []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement.lstrip().split()[0].upper())
        engine = None
        if backend == "sqlite":
            engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", capture)
        try:
            boundary, rows = await history.read_model_interaction_updates("execution", principal=_PRINCIPAL, limit=1)
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        assert boundary.cutoffs == (UsageReadCutoff("execution", 1, 2),)
        assert len(rows) == 1 and rows[0].model_request_seq == 1
        assert not rows[0].content_included and rows[0].request == {} and rows[0].response is None
        assert calls == [(_PRINCIPAL, "execution")]
        if backend == "sqlite":
            assert statements.count("SELECT") <= 5 and statements.count("BEGIN") <= 2
        _, suffix = await history.read_model_interaction_updates(
            "execution", principal=_PRINCIPAL, after_cutoffs=(UsageReadCutoff("execution", 1, 1),),
        )
        assert [item.model_request_seq for item in suffix] == [2]
        _, empty = await history.read_model_interaction_updates(
            "execution", principal=_PRINCIPAL, after_cutoffs=boundary.cutoffs,
        )
        assert empty == ()
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_update_boundary_observes_run_admitted_during_authorization(tmp_path: Path, backend: str) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        entered, proceed = asyncio.Event(), asyncio.Event()
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                entered.set()
                await proceed.wait()
        service = _service(state, authorization=Authorization())
        pending = asyncio.create_task(service.read_model_interaction_updates("execution", principal=_PRINCIPAL))
        await asyncio.wait_for(entered.wait(), 5)
        repository = state.execution.executions
        current = await repository.get("execution", tenant_id="tenant")
        await repository.compare_and_swap(
            "execution", tenant_id="tenant", expected_revision=current.revision,
            next_record=replace(current, agent_run_seq=2, revision=current.revision + 1),
        )
        await _publish(state, run_sequence=2)
        proceed.set()
        boundary, rows = await pending
        assert [value.agent_run_seq for value in boundary.cutoffs] == [1, 2]
        assert [value.agent_run_seq for value in rows] == [1, 2]
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await state.close()


@pytest.mark.parametrize("failure", ("policy", "missing", "tenant", "principal"))
async def test_update_read_does_not_disclose_metadata_without_current_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        authorized = False
        class Authorization:
            async def authorize(self, principal, action, resource):
                nonlocal authorized
                if failure == "policy" or principal.principal_id != _PRINCIPAL.principal_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                authorized = True
        repository = state.execution.executions
        get = repository.get
        async def fresh_get(*args, **kwargs):
            return None if authorized and failure == "missing" else await get(*args, **kwargs)
        async def unexpected(*args, **kwargs):
            raise AssertionError("denied metadata must not be read")
        monkeypatch.setattr(repository, "get", fresh_get)
        monkeypatch.setattr(state.run_store.read_store(RuntimeDomain.TRACE), "capture_history", unexpected)
        principal = Principal("other", "tenant") if failure == "principal" else (
            Principal("owner", "other") if failure == "tenant" else _PRINCIPAL
        )
        with pytest.raises(AIError) as raised:
            await _service(state, authorization=Authorization()).read_model_interaction_updates("execution", principal=principal)
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("kind", ("structural_service", "service_subclass", "reader", "reader_subclass", "history_subclass"))
async def test_updates_preserve_custom_public_read_dispatch(tmp_path: Path, kind: str) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        original = _service(state)
        calls = []
        class Service(DefaultExecutionHistoryService if kind == "service_subclass" else object):
            async def capture_model_interaction_cutoffs(self, execution_id, *, principal):
                calls.append("capture")
                return await original.capture_model_interaction_cutoffs(execution_id, principal=principal)
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                calls.append("metadata")
                return await original.read_model_interaction_metadata(execution_id, **kwargs)
        class Reader(StepExecutionHistoryReader if kind == "reader_subclass" else object):
            async def capture_model_interaction_cutoffs(self, execution_id, *, tenant_id):
                assert active_state_scope() is None
                calls.append("capture")
                return await _reader(state).capture_model_interaction_cutoffs(execution_id, tenant_id=tenant_id)
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                assert active_state_scope() is None
                calls.append("metadata")
                return await _reader(state).read_model_interaction_metadata(execution_id, **kwargs)
        class History(RuntimeHistory):
            async def capture_model_interaction_cutoffs(self, execution_id, **kwargs):
                calls.append("capture")
                return await super().capture_model_interaction_cutoffs(execution_id, **kwargs)
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                calls.append("metadata")
                return await super().read_model_interaction_metadata(execution_id, **kwargs)
        if kind == "structural_service":
            selected = Service()
        elif kind == "service_subclass":
            selected = Service(state.execution.executions, original._authorization, _reader(state))
        elif kind in ("reader", "reader_subclass"):
            reader = _reader(state, reader_type=Reader) if kind == "reader_subclass" else Reader()
            selected = _service(state, reader=reader)
        else:
            selected = original
        history = (History if kind == "history_subclass" else RuntimeHistory)(selected, tenant_id="tenant")
        boundary, rows = await history.read_model_interaction_updates("execution", principal=_PRINCIPAL)
        assert len(boundary.cutoffs) == len(rows) == 1
        assert calls == ["capture", "metadata"]
    finally:
        await state.close()


@pytest.mark.parametrize("phase", ("authorization", "metadata"))
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_update_cancellation_releases_owner_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    pending = None
    try:
        await _seed(state)
        entered, blocked = asyncio.Event(), asyncio.Event()
        class Authorization:
            async def authorize(self, principal, action, resource):
                assert active_state_scope() is None
                if phase == "authorization":
                    entered.set()
                    await blocked.wait()
        archive = state.run_store.read_store(RuntimeDomain.TRACE)
        original = archive.read_model_interactions_with_run
        async def read(*args, **kwargs):
            assert active_state_scope() is not None
            entered.set()
            await blocked.wait()
            return await original(*args, **kwargs)
        if phase == "metadata":
            monkeypatch.setattr(archive, "read_model_interactions_with_run", read)
        pending = asyncio.create_task(_service(state, authorization=Authorization()).read_model_interaction_updates(
            "execution", principal=_PRINCIPAL,
        ))
        await asyncio.wait_for(entered.wait(), 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert active_state_scope() is None
        monkeypatch.setattr(archive, "read_model_interactions_with_run", original)
        _, rows = await _service(state).read_model_interaction_updates("execution", principal=_PRINCIPAL)
        assert len(rows) == 1
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await state.close()


@pytest.mark.parametrize("active", (tuple(range(1, 206)), (1, 3, 205), ()))
async def test_live_update_pages_preserve_all_active_and_new_identities(tmp_path: Path, active: tuple[int, ...]) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        _, sample = await _service(state).read_model_interaction_updates("execution", principal=_PRINCIPAL)
        item = sample[0]
        requests = []
        class Service:
            async def capture_model_interaction_cutoffs(self, execution_id, *, principal):
                value = UsageReadCutoff(execution_id, 1, 206)
                return ModelInteractionReadBoundary((value,), (value,), False, True)
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                after = kwargs["after_model_request_seq"]
                through = min(kwargs["through_model_request_seq"], after + kwargs["limit"])
                requests.append((after, through))
                return tuple(replace(item, model_request_seq=n) for n in range(after + 1, through + 1))
        projection = _GraphModelProjection(RuntimeHistory(Service(), tenant_id="tenant"), _PRINCIPAL)
        await projection.add("node", project_execution_view(_execution()), 0, subscribe=False)
        projection.high_waters[("execution", 1)] = 205
        projection.active.update(("execution", 1, n) for n in active)
        values = [row.item.model_request_seq async for _, row in projection.refresh("execution")]
        assert values == [*active, 206]
        assert projection.high_waters[("execution", 1)] == 206
        assert all(0 < end - start <= 200 for start, end in requests)
        if active == tuple(range(1, 206)):
            assert requests == [(0, 200), (200, 206)]
        elif active:
            assert requests == [(0, 1), (2, 3), (204, 206)]
    finally:
        await state.close()


@pytest.mark.parametrize("optional_read", (False, True))
async def test_structural_short_pages_keep_cross_run_budget_and_finish_via_existing_pages(
    tmp_path: Path, optional_read: bool,
) -> None:
    state = _storage("memory", tmp_path)
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        _, sample = await _service(state).read_model_interaction_updates("execution", principal=_PRINCIPAL)
        requests = []
        class Service:
            async def capture_model_interaction_cutoffs(self, execution_id, *, principal):
                cutoffs = tuple(UsageReadCutoff(execution_id, run, 150) for run in (1, 2))
                return ModelInteractionReadBoundary(cutoffs, cutoffs, False, True)
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                run, after = kwargs["agent_run_seq"], kwargs["after_model_request_seq"]
                through = kwargs["through_model_request_seq"]
                requests.append((run, after, through, kwargs["limit"]))
                return tuple(replace(sample[0], agent_run_seq=run, model_request_seq=sequence)
                             for sequence in range(after + 1, min(through, after + 25) + 1))
        class PartialService(Service):
            async def read_model_interaction_updates(self, execution_id, *, principal, after_cutoffs, limit):
                boundary = await self.capture_model_interaction_cutoffs(execution_id, principal=principal)
                rows = await self.read_model_interaction_metadata(
                    execution_id, principal=principal, agent_run_seq=1,
                    after_model_request_seq=0, through_model_request_seq=150, limit=min(limit, 25),
                )
                return boundary, rows
        history = RuntimeHistory(PartialService() if optional_read else Service(), tenant_id="tenant")
        projection = _GraphModelProjection(history, _PRINCIPAL)
        await projection.add("node", project_execution_view(_execution()), 0, subscribe=False)
        rows = [value.item async for _, value in projection.refresh("execution")]
        assert [(row.agent_run_seq, row.model_request_seq) for row in rows] == [
            (run, sequence) for run in (1, 2) for sequence in range(1, 151)
        ]
        if optional_read:
            assert requests[:2] == [(1, 0, 150, 25), (1, 25, 150, 125)]
        else:
            assert requests[:2] == [(1, 0, 150, 150), (2, 0, 50, 50)]
        assert projection.high_waters == {("execution", 1): 150, ("execution", 2): 150}
        assert all(0 < limit <= 200 for _, _, _, limit in requests)
    finally:
        await state.close()
