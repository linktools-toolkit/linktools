#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Execution reservations publish their budget scope atomically."""
from ._runtime_test_helpers import model_features

from linktools.ai.persistence import Persistence
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.contracts import ExecutionStartReservationResult
from linktools.ai.persistence.state import RecordQuery, active_state_transaction
from linktools.ai.storage import InMemoryObjectStore

from . import _runtime_test_helpers
from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared_start", (False, True), ids=("atomic-start", "prepared-start"))
async def test_launch_preserves_confirmed_limits_and_inherited_usage(
    monkeypatch: pytest.MonkeyPatch, prepared_start: bool,
) -> None:
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.runtime import RuntimeFeatures

    features = RuntimeFeatures("budget-launch")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open("budget-launch", storage=PersistenceStorage.in_memory()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        service = runtime._execution_service
        backend = service._backend
        prepare_start, launch = backend.prepare_start, backend.launch
        prepared, launched = {}, {}

        async def capture_prepare(request, execution, identity):
            prepared[execution.execution_id] = request.budget
            return await prepare_start(request, execution, identity)

        async def capture_launch(request, execution):
            launched[execution.execution_id] = request.budget
            return await launch(request, execution)

        monkeypatch.setattr(backend, "prepare_start", capture_prepare)
        monkeypatch.setattr(backend, "launch", capture_launch)
        if prepared_start:
            monkeypatch.setattr(service._runtime_bridge, "prepare_reserved_start", None)
        limits = RunBudget(model_requests=1)
        root = await runtime.agents.get().start("first", budget=limits)
        assert (await root.wait()).result.status is ExecutionStatus.SUCCEEDED
        child = await root.retry("same scope")
        assert (await child.wait()).result.status is ExecutionStatus.FAILED
        assert launched == {root.execution_id: limits, child.execution_id: limits}
        if prepared_start:
            assert prepared == {root.execution_id: limits, child.execution_id: None}
        usage = await root.budget_usage()
        assert usage == await child.budget_usage()
        assert usage.model_requests == 1
        assert usage.in_flight_model_requests == usage.unknown_model_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("local_sqlite", "sqlite"))
@pytest.mark.parametrize("budget", (None, RunBudget(model_requests=2)))
async def test_caller_cancelled_reservation_completes_and_snapshot_replays(
    backend: str, budget: RunBudget | None, tmp_path: Path, monkeypatch,
) -> None:
    storage = (PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True) if backend == "local_sqlite"
               else PersistenceStorage.sqlite(tmp_path / "source.db"))
    namespace = "cancelled-budget-reservation"
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    objects = InMemoryObjectStore("snapshot")
    requests = 0
    original_model = _runtime_test_helpers._runtime_usage_model

    async def count_requests(messages, info):
        nonlocal requests
        requests += 1
        return await original_model(messages, info)

    monkeypatch.setattr(_runtime_test_helpers, "_runtime_usage_model", count_requests)
    async with Persistence.open(namespace, storage=storage) as persistence_35, Runtime.attach(persistence=persistence_35, features=model_features(RuntimeUsageModels())) as runtime:
        store = runtime._state.execution.budgets.state_store
        group = store.storage_group
        mutate = group.mutate
        committed = asyncio.Event()
        committed_reservations = []
        release = asyncio.Event()
        reservations = []
        reserve = runtime._state.execution.executions.reserve_start

        async def capture_reservation(value):
            reservations.append(value)
            return await reserve(value)

        async def pause_after_commit(stores, callback):
            outermost = active_state_transaction(store) is None
            value = await mutate(stores, callback)
            if outermost and isinstance(value, ExecutionStartReservationResult) and value.created:
                assert active_state_transaction(store) is None
                committed_reservations.append(value)
                committed.set()
                await release.wait()
            return value

        with monkeypatch.context() as patch:
            patch.setattr(runtime._state.execution.executions, "reserve_start", capture_reservation)
            patch.setattr(group, "mutate", pause_after_commit)
            start = asyncio.create_task(runtime.agents.get().start(
                "hello", budget=budget, idempotency_key="cancelled",
            ))
            try:
                await asyncio.wait_for(committed.wait(), timeout=10)
                start.cancel()

                reservation = reservations[0]
                scope = reservation.execution.budget_scope_id
                if scope is not None:
                    assert (await runtime._state.execution.budgets.read(scope)).limits == budget

                # The winner is still admitted, with its committed reply paused.
                # A concurrent idempotency loser must not create an unused scope.
                loser_id = "losing-execution"
                replay = await reserve(replace(
                    reservation,
                    execution=replace(reservation.execution, execution_id=loser_id,
                                      root_execution_id=loser_id,
                                      budget_scope_id=None if budget is None else "execution:" + loser_id),
                    idempotency=replace(reservation.idempotency, resource_id=loser_id),
                ))
                assert not replay.created
                assert replay.execution == committed_reservations[0].execution
                assert replay.execution.execution_id == reservation.execution.execution_id
                scopes = await store.read(lambda transaction: transaction.list_records(RecordQuery(kind="budget_scope")))
                assert len(scopes) == (0 if budget is None else 1)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await start
            finally:
                release.set()
                if not start.done():
                    start.cancel()
                    await asyncio.gather(start, return_exceptions=True)

        # Cancelling start only detaches its caller; the admitted owner continues.
        run = await runtime.executions.get(reservation.execution.execution_id)
        assert (await asyncio.wait_for(run.wait(), 10)).result.status is ExecutionStatus.SUCCEEDED
        assert requests == 1

    reader = (PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True) if backend == "local_sqlite"
              else PersistenceStorage.sqlite(tmp_path / "source.db"))
    reader = Persistence(reader)
    await reader.initialize(namespace=namespace, tenant_id="default", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()

    restored = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=objects, root=restored, limits=limits)
    async with Persistence.open(namespace, storage=PersistenceStorage.from_root(restored)) as persistence_102, Runtime.attach(persistence=persistence_102, features=model_features(RuntimeUsageModels())) as runtime:
        run = await runtime.agents.get().start("hello", budget=budget, idempotency_key="cancelled")
        assert run.execution_id == reservation.execution.execution_id
        assert (await run.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert requests == 1
        usage = await run.budget_usage()
        if budget is None:
            assert usage is None
        else:
            assert usage is not None
            assert usage.limits == budget
            assert usage.model_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_failed_reservation_rolls_back_budget_scope(
    backend: str, tmp_path: Path, monkeypatch,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory" else
               PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True) if backend == "local_sqlite" else
               PersistenceStorage.sqlite(tmp_path / "source.db"))
    async with Persistence.open('failed-budget-reservation', storage=storage) as persistence_124, Runtime.attach(persistence=persistence_124, features=model_features(RuntimeUsageModels())) as runtime:
        store = runtime._state.execution.budgets.state_store
        mutate = store.mutate

        async def fail_before_commit(callback):
            async def write(transaction):
                value = await callback(transaction)
                if isinstance(value, ExecutionStartReservationResult) and value.created:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return value
            return await mutate(write)

        with monkeypatch.context() as patch:
            patch.setattr(store, "mutate", fail_before_commit)
            with pytest.raises(AIError) as failed:
                await runtime.agents.get().start("hello", budget=RunBudget(model_requests=2),
                                                idempotency_key="failed")
            assert failed.value.code is ErrorCode.STORAGE_CONFLICT
        rows = await store.read(lambda transaction: transaction.scan_records())
        assert not any(row.kind in {"budget_scope", "execution", "execution_history_head", "idempotency"}
                       for row in rows)
        run = await runtime.agents.get().start("hello", budget=RunBudget(model_requests=2),
                                              idempotency_key="failed")
        assert (await run.wait()).result.status is ExecutionStatus.SUCCEEDED
