#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Already finished local work does not require another graph-wide read."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import LocalTaskGraphLauncher, TaskGraphLaunch, TaskGraphLimits, TaskNode
from linktools.ai.task._local import _GraphRun, _InflightNode

from .test_task_io_bounded_reads import _admit
from .test_task_io_measurements import _measure


class _StopAtIdle(LocalTaskGraphLauncher):
    async def _wait_scheduler(self, run, states) -> None:
        run.closed = True


async def _prepared_run(repository, *, outcome="success"):
    async def finished():
        if outcome == "failure":
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
        if outcome == "cancelled":
            raise asyncio.CancelledError

    task = asyncio.create_task(finished())
    await asyncio.gather(task, return_exceptions=True)
    run = _GraphRun(TaskGraphLaunch("graph", Principal("actor", "tenant"), TaskGraphLimits()), "worker")
    run.inflight["target"] = _InflightNode(task, None)
    return _StopAtIdle(repository, None, owner="worker"), run


async def _prepare_graph(state, width, transition):
    await _admit(state, (TaskNode("target"), TaskNode("blocker")) + tuple(
        TaskNode(f"side-{i}", ("blocker",)) for i in range(width - 2)
    ))
    repository = state.task.tasks
    await repository.claim("graph", "blocker", tenant_id="tenant", owner="other", lease_seconds=300)
    lease = await repository.claim("graph", "target", tenant_id="tenant", owner="worker", lease_seconds=300)
    if transition == "terminal":
        await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)
    else:
        await repository.bind_execution(lease, tenant_id="tenant", execution_id="execution")
        await repository.requeue_retry(lease, tenant_id="tenant", execution_id="execution", next_attempt_at=datetime.now(timezone.utc) + timedelta(hours=1))
    return repository


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ("terminal", "retry"))
async def test_completed_work_does_not_repeat_graph_reads(tmp_path, monkeypatch, transition):
    width = 3
    async with Persistence.open("completed-work", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, width, transition)
        before = await repository.graph_state("graph", tenant_id="tenant")
        launcher, run = await _prepared_run(repository)
        with _measure(monkeypatch) as counts:
            await launcher._run_graph(run)
        assert run.closed and run.failure is None and not run.inflight
        assert await repository.graph_state("graph", tenant_id="tenant") == before
        print("COMPLETED_WORK_IO " + json.dumps({"width": width, "transition": transition, "metrics": dict(counts)}, sort_keys=True))
        # A fresh, complete scheduler read is still required; a completed local
        # task must not add another full graph read or another read-only pass.
        assert counts["record_rows"] <= 2 * width + 1
        assert counts["begins"] == counts["commits"] == 1
        assert not counts["write_commits"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("failure", "cancelled"))
@pytest.mark.parametrize("transition", ("terminal", "retry"))
async def test_finished_work_preserves_failure_and_cancellation_handling(tmp_path, outcome, transition):
    async with Persistence.open("completed-outcome", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, 2, transition)
        launcher, run = await _prepared_run(repository, outcome=outcome)
        if outcome == "cancelled" and transition == "retry":
            with pytest.raises(asyncio.CancelledError):
                await launcher._run_graph(run)
            assert not run.inflight
            return
        await launcher._run_graph(run)
        assert not run.inflight
        if outcome == "failure":
            assert run.failure is not None and run.failure.code is ErrorCode.STORAGE_UNAVAILABLE
        else:
            assert run.failure is None and run.closed


@pytest.mark.asyncio
async def test_finished_work_still_requires_its_durable_node(tmp_path):
    async with Persistence.open("completed-missing-node", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, 2, "terminal")
        launcher, run = await _prepared_run(repository)

        async def delete_target(transaction):
            await transaction.delete_record(repository._state_key("graph", "target"))
            await transaction.delete_record(repository._definition_key("graph", "target"))

        await repository.state_store.mutate(delete_target)
        await launcher._run_graph(run)
        assert run.failure is not None and run.failure.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_completed_work_keeps_quiescence_started_during_read(tmp_path, monkeypatch):
    async with Persistence.open("completed-quiescence", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, 2, "terminal")
        launcher, run = await _prepared_run(repository)
        inflight = run.inflight["target"]
        original_read = repository.scheduler_state
        quiesced = asyncio.Event()

        async def quiesce():
            await asyncio.sleep(0)
            assert run.inflight.get("target") is inflight
            run.inflight.pop("target")
            quiesced.set()

        async def read(*args, **kwargs):
            current = await original_read(*args, **kwargs)
            inflight.quiesce_task = asyncio.create_task(quiesce())
            return current

        monkeypatch.setattr(repository, "scheduler_state", read)
        await launcher._run_graph(run)
        assert quiesced.is_set() and inflight.quiesce_task.done()
        assert run.closed and run.failure is None and not run.inflight


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("removed", "replaced", "already_quiescing"))
async def test_completed_work_keeps_current_inflight_owner(tmp_path, monkeypatch, change):
    async with Persistence.open("completed-owner", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, 2, "terminal")
        launcher, run = await _prepared_run(repository)
        original_read = repository.scheduler_state
        replacement_stopped = asyncio.Event()
        replacement_started = asyncio.Event()
        quiesce_release = asyncio.Event()
        owned = []

        async def replacement():
            replacement_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                replacement_stopped.set()

        async def quiesce():
            await quiesce_release.wait()
            run.inflight.pop("target")
            replacement_stopped.set()

        if change == "already_quiescing":
            pending = asyncio.create_task(quiesce())
            owned.append(pending)
            run.inflight["target"].quiesce_task = pending

        async def read(*args, **kwargs):
            current = await original_read(*args, **kwargs)
            if change == "removed":
                run.inflight.pop("target")
            elif change == "replaced":
                pending = asyncio.create_task(replacement())
                owned.append(pending)
                await replacement_started.wait()
                run.inflight["target"] = _InflightNode(pending, None)
            else:
                quiesce_release.set()
            return current

        monkeypatch.setattr(repository, "scheduler_state", read)
        try:
            await launcher._run_graph(run)
            assert run.closed and run.failure is None and not run.inflight
            assert all(task.done() for task in owned)
            if change != "removed":
                assert replacement_stopped.is_set()
        finally:
            for task in owned:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_retry_preserves_cancel_effect_started_during_read(tmp_path, monkeypatch):
    async with Persistence.open("completed-cancel-effect", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = await _prepare_graph(state, 2, "retry")
        launcher, run = await _prepared_run(repository)
        current_inflight = run.inflight["target"]
        original_read = repository.scheduler_state
        entered, release = asyncio.Event(), asyncio.Event()
        cancellations = []
        quiescence = []

        class Runner:
            async def cancel(self, invocation):
                cancellations.append(invocation.execution_id)
                entered.set()
                await release.wait()

        launcher._runner = Runner()

        async def read(*args, **kwargs):
            current = await original_read(*args, **kwargs)
            if not quiescence:
                quiescence.append(asyncio.create_task(launcher._quiesce_node(
                    run, current, "target", invoke_cancel=True,
                )))
                await entered.wait()
            return current

        monkeypatch.setattr(repository, "scheduler_state", read)
        scheduling = asyncio.create_task(launcher._run_graph(run))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert run.inflight.get("target") is current_inflight
            assert current_inflight.quiesce_task is not None
            release.set()
            await asyncio.wait_for(scheduling, 5)
            await asyncio.gather(*quiescence)
            assert cancellations == ["execution"]
            assert run.closed and run.failure is None and not run.inflight
        finally:
            release.set()
            if not scheduling.done():
                scheduling.cancel()
            await asyncio.gather(scheduling, *quiescence, return_exceptions=True)
