#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Active model projections refresh contiguous identities with bounded reads."""

from linktools.ai.persistence import Persistence
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.execution.projection import ModelProjection
from linktools.ai.execution import project_execution_view

from .test_incremental_projection_boundaries import _storage
from .test_model_capture_reads import _NAMESPACE, _PRINCIPAL, _publish, _seed, _service
from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_active_projection_refresh_preserves_all_admitted_identities(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        for sequence in range(2, 6):
            await _publish(state, request_sequence=sequence)
        service = _service(state)
        projection = ModelProjection(service, _PRINCIPAL)
        await projection.add("node", project_execution_view(_execution()), 0, subscribe=False)
        first = [row async for row in projection.refresh("execution")]
        assert [row.item.model_request_seq for _, row in first] == [1, 2, 3, 4, 5]
        statements = []
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)
        engine = None
        if backend == "sqlite":
            engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", capture)
        try:
            unchanged = [row async for row in projection.refresh("execution")]
        finally:
            if engine is not None:
                event.remove(engine, "before_cursor_execute", capture)
        assert unchanged == []
        assert len(projection.active) == 5
        assert projection.high_waters[("execution", 1)] == 5
        if backend == "sqlite":
            assert sum(statement.lstrip().upper().startswith("SELECT") for statement in statements) <= 8
        await projection.close()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("active,previous,current,expected", (
    ((1, 3, 5), 5, 6, ((0, 1), (2, 3), (4, 6))),
    (tuple(range(1, 206)), 205, 205, ((0, 200), (200, 205))),
    ((), 5, 6, ((5, 6),)),
))
async def test_refresh_keeps_sparse_ranges_and_bounded_pages(
    active: tuple[int, ...], previous: int, current: int,
    expected: tuple[tuple[int, int], ...],
) -> None:
    from dataclasses import replace
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.execution import ModelInteractionReadBoundary, UsageReadCutoff

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        item, = await _service(state).read_model_interaction_metadata(
            "execution", principal=_PRINCIPAL, agent_run_seq=1,
            after_model_request_seq=0, through_model_request_seq=1,
        )
        requested = []
        class History:
            async def read_model_interaction_metadata(self, execution_id, **kwargs):
                assert execution_id == "execution" and kwargs["principal"] is _PRINCIPAL
                after = kwargs["after_model_request_seq"]
                through = min(kwargs["through_model_request_seq"], after + kwargs["limit"])
                requested.append((after, through))
                return tuple(replace(item, model_request_seq=n) for n in range(after + 1, through + 1))
        projection = ModelProjection(History(), _PRINCIPAL)
        await projection.add("node", project_execution_view(_execution()), 0, subscribe=False)
        projection.high_waters[("execution", 1)] = previous
        projection.active.update(("execution", 1, n) for n in active)
        cutoff = UsageReadCutoff("execution", 1, current)
        boundary = ModelInteractionReadBoundary((cutoff,), (cutoff,), False, True)
        result = [row async for row in projection.refresh("execution", boundary=boundary)]
        assert requested == list(expected)
        assert [row.item.model_request_seq for _, row in result] == [
            n for start, end in expected for n in range(start + 1, end + 1)
        ]
        await projection.close()
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("denied", "missing", "wrong_owner"))
@pytest.mark.parametrize("composed", (False, True))
async def test_batched_refresh_propagates_authority_and_integrity_failures(failure: str, composed: bool) -> None:
    from dataclasses import replace
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence import PersistenceStorage

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        await _publish(state, request_sequence=2)
        service = _service(state)
        projection = ModelProjection(service, _PRINCIPAL)
        await projection.add("node", project_execution_view(_execution()), 0, subscribe=False)
        assert len([row async for row in projection.refresh("execution")]) == 2
        original = service.read_model_interaction_updates if composed else service.read_model_interaction_metadata
        async def read(*args, **kwargs):
            if failure == "denied":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            result = await original(*args, **kwargs)
            boundary, rows = result if composed else (None, result)
            rows = () if failure == "missing" else (replace(rows[0], execution_id="another"), *rows[1:])
            return (boundary, rows) if composed else rows
        if composed:
            service.read_model_interaction_updates = read
        else:
            service.read_model_interaction_updates = None
            service.read_model_interaction_metadata = read
        with pytest.raises(AIError) as raised:
            _ = [row async for row in projection.refresh("execution")]
        assert raised.value.code is (
            ErrorCode.AUTHORIZATION_DENIED if failure == "denied" else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
        await projection.close()
    finally:
        await state.close()
