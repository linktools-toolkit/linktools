#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared budget admission, durable usage, and backend parity."""

import asyncio
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from linktools.ai.core import BudgetUsage, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.runtime.state._budget import BudgetRepositoryImpl
from linktools.ai.runtime.state._snapshot_validation import (
    canonical_snapshot_indexes, validate_snapshot_domain,
)
from linktools.ai.runtime.state._sql import _SqlTransaction


@pytest_asyncio.fixture(params=("memory", "filesystem", "sqlite"))
async def budget_storage(request, tmp_path):
    if request.param == "memory":
        storage = RuntimeStorage.in_memory()
    elif request.param == "filesystem":
        storage = RuntimeStorage.filesystem(tmp_path / "runtime")
    else:
        storage = RuntimeStorage.sqlite(tmp_path / "runtime.db")
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.parametrize("kwargs", [
    {"model_requests": -1}, {"tool_calls": True}, {"total_tokens": 1.5},
    {"deadline_at": datetime(2026, 1, 1)},
])
def test_budget_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        RunBudget(**kwargs)


def test_budget_is_frozen_and_deadline_identity_uses_utc():
    budget = RunBudget(model_requests=0, tool_calls=0, total_tokens=0,
                       deadline_at=datetime(2026, 1, 1, 8, tzinfo=timezone(timedelta(hours=8))))
    with pytest.raises(FrozenInstanceError):
        budget.model_requests = 2
    assert budget.digest_payload()["deadline_at"] == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_budget_scope_is_immutable_and_missing_scope_fails_closed(budget_storage):
    budgets = budget_storage.execution.budgets
    limits = RunBudget(model_requests=0, tool_calls=0)
    usage = await budgets.ensure("scope", limits)
    assert await budgets.ensure("scope", limits) == usage
    with pytest.raises(AIError) as conflict:
        await budgets.ensure("scope", RunBudget(model_requests=1))
    assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    with pytest.raises(AIError) as missing:
        await budgets.admit_model("missing", "request")
    assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    for admission in (budgets.admit_model, budgets.admit_tool):
        with pytest.raises(AIError) as exhausted:
            await admission("scope", "call")
        assert exhausted.value.safe_details["reason"] == "limit_exceeded"
    assert await budgets.read("scope") == usage


@pytest.mark.asyncio
async def test_budget_counts_are_atomic_across_concurrent_callers(budget_storage):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(model_requests=3, tool_calls=2))
    results = await asyncio.gather(
        *(budgets.admit_model("scope", f"request-{index}") for index in range(8)),
        *(budgets.admit_tool("scope", f"tool-{index}") for index in range(8)),
        return_exceptions=True,
    )
    assert sum(isinstance(value, BudgetUsage) for value in results) == 5
    assert all(isinstance(value, BudgetUsage) or (
        isinstance(value, AIError) and value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    ) for value in results)
    usage = await budgets.read("scope")
    assert (usage.model_requests, usage.tool_calls, usage.in_flight_model_requests) == (3, 2, 3)


@pytest.mark.asyncio
async def test_soft_token_threshold_preserves_concurrency_and_observes_actual_usage(budget_storage):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=10))
    await asyncio.gather(*(budgets.admit_model("scope", str(index)) for index in range(3)))
    assert (await budgets.read("scope")).in_flight_model_requests == 3
    await budgets.settle_model("scope", "0", 4)
    await budgets.settle_model("scope", "1", 4)
    await budgets.admit_model("scope", "3")
    await budgets.settle_model("scope", "2", 5)
    for action in (lambda: budgets.admit_model("scope", "4"),
                   lambda: budgets.admit_tool("scope", "tool"),
                   lambda: budgets.check("scope")):
        with pytest.raises(AIError) as exhausted:
            await action()
        assert exhausted.value.safe_details["dimension"] == "total_tokens"
        assert exhausted.value.safe_details["reason"] == "limit_exceeded"
    usage = await budgets.settle_model("scope", "3", 6)
    assert (usage.total_tokens, usage.in_flight_model_requests) == (19, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ("check", "model", "tool"))
@pytest.mark.parametrize("outcome", ("known", "unknown", "cancelled"))
async def test_sql_admission_rechecks_local_receipt_settled_after_snapshot(
    action, outcome, tmp_path, monkeypatch,
):
    database = tmp_path / "budget.db"
    storage = RuntimeStorage.sqlite(database)
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
    release = asyncio.Event()
    admission = None
    try:
        budgets = storage.execution.budgets
        await budgets.ensure("scope", RunBudget(total_tokens=100))
        await budgets.admit_model("scope", "first")
        if action != "check":
            # Server SQL dialects allow simultaneous mutations. Exercise that
            # path with real SQLite transactions rather than its local guard.
            monkeypatch.setattr(budgets.state_store.storage_group, "_mutation_lock", None)
        observed = asyncio.Event()
        original_list = _SqlTransaction.list_records

        async def pause_after_receipts(transaction, query):
            rows = await original_list(transaction, query)
            if query.kind == "budget_model" and not observed.is_set():
                observed.set()
                await release.wait()
            return rows

        monkeypatch.setattr(_SqlTransaction, "list_records", pause_after_receipts)
        operation = (budgets.check("scope") if action == "check" else
                     budgets.admit_model("scope", "next") if action == "model" else
                     budgets.admit_tool("scope", "tool"))
        admission = asyncio.create_task(operation)
        await asyncio.wait_for(observed.wait(), timeout=2)
        if outcome == "cancelled":
            committed, release_settlement = asyncio.Event(), asyncio.Event()
            original_mutate = budgets.state_store.mutate

            async def pause_after_commit(callback):
                value = await original_mutate(callback)
                if not committed.is_set():
                    committed.set()
                    await release_settlement.wait()
                return value

            monkeypatch.setattr(budgets.state_store, "mutate", pause_after_commit)
            settlement = asyncio.create_task(budgets.settle_model("scope", "first", 3))
            try:
                await asyncio.wait_for(committed.wait(), timeout=2)
                settlement.cancel()
            finally:
                release_settlement.set()
            with pytest.raises(asyncio.CancelledError):
                await settlement
        else:
            await asyncio.wait_for(
                budgets.settle_model("scope", "first", None if outcome == "unknown" else 3),
                timeout=2,
            )
        release.set()
        if outcome == "unknown":
            with pytest.raises(AIError) as unknown:
                await asyncio.wait_for(admission, timeout=2)
            assert unknown.value.safe_details["reason"] == "unknown_usage"
        else:
            usage = await asyncio.wait_for(admission, timeout=2)
            assert usage.total_tokens == 3
        usage = await budgets.read("scope")
        admitted = outcome != "unknown"
        assert usage.model_requests == 1 + int(admitted and action == "model")
        assert usage.tool_calls == int(admitted and action == "tool")
        assert usage.in_flight_model_requests == int(admitted and action == "model")
        assert usage.unknown_model_requests == int(outcome == "unknown")
    finally:
        release.set()
        if admission is not None:
            await asyncio.gather(admission, return_exceptions=True)
        await storage.close()


@pytest.mark.asyncio
async def test_failed_settlement_leaves_same_runtime_reservation_fail_closed(budget_storage, monkeypatch):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=100))
    await budgets.admit_model("scope", "first")
    original_mutate = budgets.state_store.mutate

    async def abort_settlement(callback):
        async def abort(transaction):
            await callback(transaction)
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        return await original_mutate(abort)

    with monkeypatch.context() as patch:
        patch.setattr(budgets.state_store, "mutate", abort_settlement)
        with pytest.raises(AIError) as failed:
            await budgets.settle_model("scope", "first", 3)
        assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
    for action in (lambda: budgets.check("scope"),
                   lambda: budgets.admit_model("scope", "next"),
                   lambda: budgets.admit_tool("scope", "tool")):
        with pytest.raises(AIError) as abandoned:
            await asyncio.wait_for(action(), timeout=2)
        assert abandoned.value.safe_details["reason"] == "unresolved_in_flight"
    usage = await budgets.read("scope")
    assert (usage.model_requests, usage.tool_calls, usage.total_tokens) == (1, 0, 0)
    assert (usage.in_flight_model_requests, usage.unknown_model_requests) == (1, 0)


@pytest.mark.asyncio
async def test_dispatch_identity_cannot_be_readmitted_and_settlement_is_idempotent(budget_storage):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget())
    await budgets.admit_model("scope", "model")
    await budgets.admit_tool("scope", "tool")
    for admission, identity in ((budgets.admit_model, "model"), (budgets.admit_tool, "tool")):
        with pytest.raises(AIError) as duplicate:
            await admission("scope", identity)
        assert duplicate.value.safe_details["reason"] == "duplicate_dispatch"
    usage = await budgets.settle_model("scope", "model", 7)
    assert await budgets.settle_model("scope", "model", 7) == usage
    with pytest.raises(AIError) as duplicate:
        await budgets.admit_model("scope", "model")
    assert duplicate.value.safe_details["reason"] == "duplicate_dispatch"
    with pytest.raises(AIError) as conflict:
        await budgets.settle_model("scope", "model", 8)
    assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
    assert (await budgets.read("scope")).total_tokens == 7


@pytest.mark.asyncio
async def test_unknown_usage_blocks_token_limits_but_not_count_only_limits(budget_storage):
    budgets = budget_storage.execution.budgets
    for scope, limits in (("tokens", RunBudget(total_tokens=20)),
                          ("counts", RunBudget(model_requests=2, tool_calls=1))):
        await budgets.ensure(scope, limits)
        await budgets.admit_model(scope, "first")
        usage = await budgets.settle_model(scope, "first", None)
        assert (usage.total_tokens, usage.in_flight_model_requests, usage.unknown_model_requests) == (0, 0, 1)
    for action in (lambda: budgets.admit_model("tokens", "next"),
                   lambda: budgets.admit_tool("tokens", "tool"),
                   lambda: budgets.check("tokens")):
        with pytest.raises(AIError) as unknown:
            await action()
        assert unknown.value.safe_details["reason"] == "unknown_usage"
    await budgets.admit_model("counts", "next")
    await budgets.admit_tool("counts", "tool")
    assert (await budgets.read("counts")).model_requests == 2


@pytest.mark.asyncio
async def test_foreign_runtime_inflight_usage_fails_closed_without_waiting(budget_storage):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=20))
    await budgets.admit_model("scope", "first")
    other = BudgetRepositoryImpl(budgets.state_store, namespace="budget-tests", tenant_id="tenant")
    with pytest.raises(AIError) as unknown:
        await asyncio.wait_for(other.admit_model("scope", "second"), timeout=1)
    assert unknown.value.safe_details["reason"] == "unresolved_in_flight"
    assert (await other.read("scope")).in_flight_model_requests == 1
    await budgets.settle_model("scope", "first", 3)
    assert (await other.admit_model("scope", "second")).model_requests == 2


@pytest.mark.asyncio
async def test_deadline_gates_new_dispatch_but_does_not_prevent_settlement(budget_storage, monkeypatch):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("expired", RunBudget(deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
    for action in (lambda: budgets.admit_model("expired", "m"),
                   lambda: budgets.admit_tool("expired", "t"),
                   lambda: budgets.check("expired")):
        with pytest.raises(AIError) as expired:
            await action()
        assert expired.value.safe_details["reason"] == "deadline_exceeded"
    await budgets.ensure("settle", RunBudget(deadline_at=datetime.now(timezone.utc) + timedelta(seconds=1)))
    await budgets.admit_model("settle", "m")

    class ExpiredClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(seconds=2)

    monkeypatch.setattr("linktools.ai.runtime.state._budget.datetime", ExpiredClock)
    with pytest.raises(AIError) as expired:
        await budgets.admit_model("settle", "next")
    assert expired.value.safe_details["reason"] == "deadline_exceeded"
    assert (await budgets.settle_model("settle", "m", 2)).total_tokens == 2


@pytest.mark.asyncio
async def test_budget_snapshot_validates_reservation_projection(budget_storage):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=20))
    await budgets.admit_model("scope", "m")
    await budgets.settle_model("scope", "m", 5)
    await budgets.admit_tool("scope", "t")
    rows = await budgets.state_store.read(lambda transaction: transaction.scan_records())
    args = dict(namespace="budget-tests", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
                records=rows, facts=(), operations=())
    aliases, sequences = canonical_snapshot_indexes(**args)
    validate_snapshot_domain(**args, aliases=aliases, sequences=sequences)
    usage = await budgets.read("scope")
    corrupt = budgets._stored("budget_scope", "scope", replace(usage, total_tokens=6))
    args["records"] = tuple(corrupt if row.kind == "budget_scope" else row for row in rows)
    with pytest.raises(AIError) as integrity:
        validate_snapshot_domain(**args, aliases=aliases, sequences=sequences)
    assert integrity.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_cancelled_admission_retains_count_and_marks_unknown(monkeypatch):
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        budgets = storage.execution.budgets
        await budgets.ensure("scope", RunBudget(total_tokens=20))
        committed = asyncio.Event()
        release = asyncio.Event()
        original = budgets.state_store.mutate

        async def pause_after_commit(callback):
            value = await original(callback)
            if not committed.is_set():
                committed.set()
                await release.wait()
            return value

        monkeypatch.setattr(budgets.state_store, "mutate", pause_after_commit)
        admission = asyncio.create_task(budgets.admit_model("scope", "m"))
        await committed.wait()
        admission.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await admission
        usage = await budgets.read("scope")
        assert (usage.model_requests, usage.in_flight_model_requests, usage.unknown_model_requests) == (1, 0, 1)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_commit_readback_recovers_own_admission_without_granting_duplicate(monkeypatch):
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        budgets = storage.execution.budgets
        await budgets.ensure("scope", RunBudget(model_requests=2))
        original = budgets.state_store.mutate

        async def lose_commit_response(callback):
            await original(callback)
            raise RuntimeError("response lost after commit")

        monkeypatch.setattr(budgets.state_store, "mutate", lose_commit_response)
        assert (await budgets.admit_model("scope", "m")).model_requests == 1
        with pytest.raises(AIError) as duplicate:
            await budgets.admit_model("scope", "m")
        assert duplicate.value.safe_details["reason"] == "duplicate_dispatch"
        assert (await budgets.settle_model("scope", "m", 3)).total_tokens == 3
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_budget_reopen_retains_counts_unknown_and_dispatch_identities(backend, tmp_path):
    def open_storage():
        return (RuntimeStorage.filesystem(tmp_path / "state") if backend == "filesystem"
                else RuntimeStorage.sqlite(tmp_path / "state.db"))

    storage = open_storage()
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    budgets = storage.execution.budgets
    await budgets.ensure("tokens", RunBudget(total_tokens=100))
    await budgets.admit_model("tokens", "pending")
    await budgets.ensure("counts", RunBudget(model_requests=2, tool_calls=2))
    await budgets.admit_model("counts", "done")
    await budgets.settle_model("counts", "done", 7)
    await budgets.admit_tool("counts", "tool")
    await storage.close()
    storage = open_storage()
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        budgets = storage.execution.budgets
        with pytest.raises(AIError) as unknown:
            await budgets.admit_model("tokens", "next")
        assert unknown.value.safe_details["reason"] == "unresolved_in_flight"
        with pytest.raises(AIError) as duplicate:
            await budgets.admit_tool("counts", "tool")
        assert duplicate.value.safe_details["reason"] == "duplicate_dispatch"
        await budgets.settle_model("counts", "done", 7)
        await budgets.admit_model("counts", "next")
        usage = await budgets.read("counts")
        assert (usage.model_requests, usage.tool_calls, usage.total_tokens) == (2, 1, 7)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_shared_budget_counts_across_independent_sqlite_instances(tmp_path):
    def open_storage():
        return RuntimeStorage.sqlite(tmp_path / "state.db")

    first, second = open_storage(), open_storage()
    await first.initialize(namespace="budget-tests", tenant_id="tenant")
    await second.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        await first.execution.budgets.ensure("scope", RunBudget(tool_calls=3))
        results = await asyncio.gather(*(
            (first if index % 2 else second).execution.budgets.admit_tool("scope", str(index))
            for index in range(12)
        ), return_exceptions=True)
        assert sum(isinstance(value, BudgetUsage) for value in results) == 3
        assert all(isinstance(value, BudgetUsage) or (
            isinstance(value, AIError) and value.code is ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
        ) for value in results)
        assert (await second.execution.budgets.read("scope")).tool_calls == 3
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_storage_close_drains_cancelled_budget_commit_before_releasing_backend(tmp_path, monkeypatch):
    path = tmp_path / "state"
    storage = RuntimeStorage.filesystem(path)
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    budgets = storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=20))
    committed, release = asyncio.Event(), asyncio.Event()
    original = budgets.state_store.mutate

    async def pause_after_commit(callback):
        value = await original(callback)
        if not committed.is_set():
            committed.set()
            await release.wait()
        return value

    monkeypatch.setattr(budgets.state_store, "mutate", pause_after_commit)
    admission = asyncio.create_task(budgets.admit_model("scope", "m"))
    await committed.wait()
    admission.cancel()
    closing = asyncio.create_task(storage.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await admission
    await closing
    with pytest.raises(AIError) as closed:
        await budgets.admit_model("scope", "next")
    assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    reopened = RuntimeStorage.filesystem(path)
    await reopened.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        usage = await reopened.execution.budgets.read("scope")
        assert (usage.model_requests, usage.unknown_model_requests) == (1, 1)
    finally:
        await reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_budget_reservations_survive_portable_snapshot(backend, tmp_path):
    def open_storage():
        return (RuntimeStorage.filesystem(tmp_path / "state") if backend == "filesystem"
                else RuntimeStorage.sqlite(tmp_path / "state.db"))

    storage = open_storage()
    await storage.initialize(namespace="budget-tests", tenant_id="tenant")
    budgets = storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=30))
    await budgets.admit_model("scope", "done")
    await budgets.settle_model("scope", "done", 7)
    await budgets.admit_tool("scope", "tool")
    await budgets.admit_model("scope", "pending")
    expected = await budgets.read("scope")
    await storage.close()
    reader = open_storage()
    await reader.initialize(namespace="budget-tests", tenant_id="tenant", read_only=True)
    objects = InMemoryObjectStore("budget-snapshot")
    limits = SnapshotLimits(max_entries=100, max_bytes=1024 * 1024)
    try:
        reference = await reader.export_snapshot(object_store=objects, limits=limits)
    finally:
        await reader.close()
    root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(reference, object_store=objects, root=root, limits=limits)
    restored = RuntimeStorage.from_root(root)
    await restored.initialize(namespace="budget-tests", tenant_id="tenant")
    try:
        assert await restored.execution.budgets.read("scope") == expected
        with pytest.raises(AIError) as unknown:
            await restored.execution.budgets.admit_model("scope", "next")
        assert unknown.value.safe_details["reason"] == "unresolved_in_flight"
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", (True, False))
async def test_admission_revalidates_mixed_revision_receipts_before_integrity_failure(
    budget_storage, monkeypatch, settled,
):
    budgets = budget_storage.execution.budgets
    await budgets.ensure("scope", RunBudget(total_tokens=100))
    await budgets.admit_model("scope", "first")
    observed = await budgets.state_store.read(lambda tx: budgets._scope_record(tx, "scope"))
    if settled:
        await budgets.settle_model("scope", "first", 3)
    original = budgets._scope_record
    first = True

    async def mixed_observation(transaction, scope_id):
        nonlocal first
        if first:
            first = False
            if settled:
                return observed
            row, usage = observed
            return row, replace(usage, in_flight_model_requests=2, model_requests=2)
        return await original(transaction, scope_id)

    monkeypatch.setattr(budgets, "_scope_record", mixed_observation)
    if settled:
        usage = await budgets.admit_tool("scope", "tool")
        assert (usage.total_tokens, usage.tool_calls) == (3, 1)
    else:
        with pytest.raises(AIError) as corrupt:
            await budgets.admit_tool("scope", "tool")
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
