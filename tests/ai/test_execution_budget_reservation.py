#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution reservations publish their budget scope atomically."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.runtime.state._contracts import ExecutionStartReservationResult
from linktools.ai.runtime.state._store import RecordQuery
from linktools.ai.storage import InMemoryObjectStore

from ._runtime_test_helpers import RuntimeUsageModels


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("budget", (None, RunBudget(model_requests=2)))
async def test_cancelled_reservation_snapshot_restores_and_replays(
    backend: str, budget: RunBudget | None, tmp_path: Path, monkeypatch,
) -> None:
    storage = (RuntimeStorage.filesystem(tmp_path / "source") if backend == "filesystem"
               else RuntimeStorage.sqlite(tmp_path / "source.db"))
    namespace = "cancelled-budget-reservation"
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    objects = InMemoryObjectStore("snapshot")
    async with Runtime.open(namespace, models=RuntimeUsageModels(), storage=storage) as runtime:
        store = storage.execution.budgets.state_store
        mutate = store.mutate
        committed = asyncio.Event()
        release = asyncio.Event()
        reservations = []
        reserve = storage.execution.executions.reserve_start

        async def capture_reservation(value):
            reservations.append(value)
            return await reserve(value)

        async def pause_after_commit(callback):
            value = await mutate(callback)
            if isinstance(value, ExecutionStartReservationResult):
                committed.set()
                await release.wait()
            return value

        with monkeypatch.context() as patch:
            patch.setattr(storage.execution.executions, "reserve_start", capture_reservation)
            patch.setattr(store, "mutate", pause_after_commit)
            start = asyncio.create_task(runtime.agents.get().start(
                "hello", budget=budget, idempotency_key="cancelled",
            ))
            try:
                await asyncio.wait_for(committed.wait(), timeout=10)
                start.cancel()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await start
            finally:
                release.set()
                if not start.done():
                    start.cancel()
                    await asyncio.gather(start, return_exceptions=True)

        reservation = reservations[0]
        scope = reservation.execution.budget_scope_id
        if scope is not None:
            assert (await storage.execution.budgets.read(scope)).limits == budget

        # A concurrent idempotency loser must not create its own unused scope.
        loser_id = "losing-execution"
        replay = await reserve(replace(
            reservation,
            execution=replace(reservation.execution, execution_id=loser_id,
                              root_execution_id=loser_id,
                              budget_scope_id=None if budget is None else "execution:" + loser_id),
            idempotency=replace(reservation.idempotency, resource_id=loser_id),
        ))
        assert not replay.created
        assert replay.execution == reservation.execution
        scopes = await store.read(lambda transaction: transaction.list_records(RecordQuery(kind="budget_scope")))
        assert len(scopes) == (0 if budget is None else 1)

    reader = (RuntimeStorage.filesystem(tmp_path / "source") if backend == "filesystem"
              else RuntimeStorage.sqlite(tmp_path / "source.db"))
    await reader.initialize(namespace=namespace, tenant_id="default", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()

    restored = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(reference, object_store=objects, root=restored, limits=limits)
    async with Runtime.open(namespace, models=RuntimeUsageModels(),
                            storage=RuntimeStorage.from_root(restored)) as runtime:
        run = await runtime.agents.get().start("hello", budget=budget, idempotency_key="cancelled")
        assert run.execution_id == reservation.execution.execution_id
        assert (await run.wait()).result.status is ExecutionStatus.SUCCEEDED
        usage = await run.budget_usage()
        if budget is None:
            assert usage is None
        else:
            assert usage is not None
            assert usage.limits == budget
            assert usage.model_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_failed_reservation_rolls_back_budget_scope(
    backend: str, tmp_path: Path, monkeypatch,
) -> None:
    storage = (RuntimeStorage.in_memory() if backend == "memory" else
               RuntimeStorage.filesystem(tmp_path / "source") if backend == "filesystem" else
               RuntimeStorage.sqlite(tmp_path / "source.db"))
    async with Runtime.open("failed-budget-reservation", models=RuntimeUsageModels(), storage=storage) as runtime:
        store = storage.execution.budgets.state_store
        mutate = store.mutate

        async def fail_before_commit(callback):
            async def write(transaction):
                value = await callback(transaction)
                if isinstance(value, ExecutionStartReservationResult):
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
