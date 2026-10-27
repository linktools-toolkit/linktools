#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed evaluation coordinators reject new work without changing durable facts."""

from linktools.ai.persistence import Persistence
from linktools.ai.task.engine import TaskEngine
from linktools.ai.evaluation.engine import Evaluations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus, idempotency_key_digest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationSpec,
    HumanScoreRequest, RescoreRequest, ScoreAttemptView, ScoreBundle, ScorerSpec,
    StartEvaluationRequest,
)
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.persistence.contracts import EvaluationRecord
from linktools.ai.task import Task, TaskRef

from .test_evaluation_consumers import CONTEXT, DIMENSION, PRINCIPAL, echo


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("start", "rescore", "cancel", "reconcile", "human_score"))
async def test_closed_evaluations_reject_control_without_persisting(
    tmp_path: Path, operation: str,
) -> None:
    target = Task("lifecycle.target", echo, effect_policy="none")
    scorer = ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,))
    storage = PersistenceStorage.filesystem(tmp_path)
    async with (
        Persistence.open('evaluation-lifecycle', storage=storage, tenant_id='evaluation-consumers') as persistence_34,
        TaskEngine.open(target, persistence=persistence_34, context=CONTEXT) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("lifecycle", 1), (
            CaseSpec.task(CaseRef("lifecycle", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("candidate", task=target.ref),), (scorer,)), PRINCIPAL, "start")
        run = await evaluations.start(request)

        async def waiting_score() -> ScoreAttemptView:
            while True:
                pending = (await run.scores()).items[0]
                if pending.scorer_execution is not None:
                    graph = await engine.get(pending.scorer_graph.graph_id, principal=PRINCIPAL)
                    state = await graph.state()
                    node = next(item for item in state.node_states if item.node_id == pending.scorer_node_id)
                    if node.status is TaskStatus.WAITING:
                        return pending
                await asyncio.sleep(0.01)

        pending = await asyncio.wait_for(waiting_score(), 10)
        await evaluations.close()
        repository = persistence_34.evaluation.records
        before = await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation")))
        assert len(before) == 1
        key = "closed-request"
        with pytest.raises(AIError) as raised:
            if operation == "start":
                await evaluations.start(replace(request, idempotency_key=key))
            elif operation == "rescore":
                await run.rescore(RescoreRequest((scorer,), key))
            elif operation == "cancel":
                await run.cancel(idempotency_key=key)
            elif operation == "reconcile":
                await evaluations.reconcile(run.experiment_id,
                                                    principal=PRINCIPAL, idempotency_key=key)
            else:
                await run.submit_human_score(HumanScoreRequest(pending.trial.trial_id,
                    pending.scorer_slot_id, pending.evidence_ref,
                    ScoreBundle(dimensions={"exact_match": 1.0}), key))
        assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert not raised.value.retryable
        assert await repository.state_store.read(
            lambda transaction: transaction.list_records(RecordQuery(kind="evaluation"))) == before
        assert await persistence_34.evaluation.idempotency.get("evaluation.run", idempotency_key_digest(key),
                                                       tenant_id=PRINCIPAL.tenant_id) is None
        reopened = await evaluations.get(run.experiment_id, principal=PRINCIPAL)
        assert (await reopened.inspect()).completion == "running"
        assert (await reopened.scores()).items == (pending,)
        assert (await reopened.trials()).items[0].evidence_ref is not None
        assert (await evaluations.get_dataset(dataset, principal=PRINCIPAL)).ref == dataset


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", (False, True))
async def test_close_waits_for_unwatched_admission_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_start: bool,
) -> None:
    target = Task("settling.target", echo, effect_policy="none")
    async with (
        Persistence.open("settling", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
                         tenant_id=PRINCIPAL.tenant_id) as persistence,
        TaskEngine.open(target, persistence=persistence) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        evaluations.preflight_handoff()
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("settling", 1), (
            CaseSpec.task(CaseRef("settling", "one", 1), input={"answer": "yes"}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        request = StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("target", task=target.ref),),
            (ScorerSpec("human", TaskRef.deferred_input(), (DIMENSION,)),)), PRINCIPAL, "start")
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        reserve = persistence.evaluation.records.reserve_experiment

        async def settle(record: EvaluationRecord) -> EvaluationRecord:
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                await reserve(record)
                raise
            return await reserve(record)

        monkeypatch.setattr(persistence.evaluation.records, "reserve_experiment", settle)
        starting = asyncio.create_task(evaluations.start(request))
        await asyncio.wait_for(entered.wait(), 10)
        with pytest.raises(AIError) as rejected:
            evaluations.preflight_handoff()
        assert rejected.value.safe_details["reason"] == "active_evaluation"
        # A preflight failure does not close otherwise usable domain operations.
        assert (await evaluations.get_dataset(dataset, principal=PRINCIPAL)).ref == dataset
        evaluations.stop_admission()
        with pytest.raises(AIError):
            await evaluations.start(replace(request, idempotency_key="late"))
        if cancel_start:
            starting.cancel()
            await asyncio.wait_for(cancelled.wait(), 10)
        closing = asyncio.create_task(evaluations.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        retry = asyncio.create_task(evaluations.close())
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        if cancel_start:
            with pytest.raises(asyncio.CancelledError):
                await starting
        else:
            await starting
        await asyncio.wait_for(retry, 10)
        assert persistence.ready
        assert await persistence.evaluation.idempotency.get(
            "evaluation.run", idempotency_key_digest("start"), tenant_id=PRINCIPAL.tenant_id,
        ) is not None
        evaluations.preflight_handoff()
        await evaluations.close()


@pytest.mark.asyncio
async def test_failed_scope_exit_remains_blocked_without_reentering_or_closing_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import AsyncIterator
    from contextlib import asynccontextmanager
    from linktools.ai.evaluation.engine._scope import _EnteredTrialScope

    exits: list[str] = []
    reader_closes: list[str] = []
    async with (
        Persistence.open("failed-scope", storage=PersistenceStorage.in_memory()) as persistence,
        TaskEngine.open(persistence=persistence) as engine,
    ):
        manager = Evaluations.open(engine)
        evaluations = await manager.__aenter__()
        original_close = evaluations._history.close

        async def close_reader() -> None:
            reader_closes.append("reader")
            await original_close()

        monkeypatch.setattr(evaluations._history, "close", close_reader)

        @asynccontextmanager
        async def scope() -> AsyncIterator[TaskEngine[None]]:
            try:
                yield engine
            finally:
                exits.append("exit")
                raise ValueError("scope exit failed")

        entered = _EnteredTrialScope(scope())
        await entered.engine()
        evaluations._trial_scopes[("experiment", "slot")] = entered
        try:
            for _ in range(2):
                with pytest.raises(ValueError, match="scope exit failed"):
                    await evaluations.close()
            assert exits == ["exit"]
            assert reader_closes == []
            assert persistence.ready
            with pytest.raises(AIError):
                evaluations.preflight_handoff()
        finally:
            # The injected scope has no resources; remove it only to release this fixture.
            evaluations._trial_scopes.clear()
            await manager.__aexit__(None, None, None)
        assert reader_closes
        assert persistence.ready


@pytest.mark.asyncio
async def test_context_exit_retains_primary_and_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with (
        Persistence.open("failed-body", storage=PersistenceStorage.in_memory()) as persistence,
        TaskEngine.open(persistence=persistence) as engine,
    ):
        manager = Evaluations.open(engine)
        evaluations = await manager.__aenter__()
        original_close = evaluations._history.close
        primary = asyncio.CancelledError("body cancelled")
        cleanup = ValueError("reader close failed")

        async def fail_reader() -> None:
            raise cleanup

        monkeypatch.setattr(evaluations._history, "close", fail_reader)
        with pytest.raises(BaseExceptionGroup) as raised:
            await manager.__aexit__(type(primary), primary, primary.__traceback__)
        assert raised.value.exceptions == (primary, cleanup)
        assert persistence.ready
        monkeypatch.setattr(evaluations._history, "close", original_close)
        await evaluations.close()
        assert persistence.ready


@pytest.mark.asyncio
async def test_observation_cleanup_cancellation_cannot_claim_closed() -> None:
    class InterruptedObservation:
        interrupted = False

        def stop(self) -> None:
            pass

        async def close(self) -> None:
            if not self.interrupted:
                self.interrupted = True
                raise asyncio.CancelledError("observation still closing")

    async with (
        Persistence.open("observation-close", storage=PersistenceStorage.in_memory()) as persistence,
        TaskEngine.open(persistence=persistence) as engine,
        Evaluations.open(engine) as evaluations,
    ):
        observation = InterruptedObservation()
        evaluations._observation_sessions.add(observation)
        with pytest.raises(asyncio.CancelledError):
            await evaluations.close()
        assert not evaluations._closed
        assert persistence.ready
        await evaluations.close()
        assert evaluations._closed


@pytest.mark.asyncio
@pytest.mark.parametrize("body_failed", (False, True))
async def test_context_exit_cancellation_retains_cleanup_failure_and_body_error(
    monkeypatch: pytest.MonkeyPatch, body_failed: bool,
) -> None:
    async with (
        Persistence.open("cancelled-exit", storage=PersistenceStorage.in_memory()) as persistence,
        TaskEngine.open(persistence=persistence) as engine,
    ):
        manager = Evaluations.open(engine)
        evaluations = await manager.__aenter__()
        original_close = evaluations._history.close
        entered, release = asyncio.Event(), asyncio.Event()
        primary = ValueError("body failed") if body_failed else None
        cleanup = RuntimeError("reader cleanup failed")

        async def fail_reader() -> None:
            entered.set()
            await release.wait()
            raise cleanup

        monkeypatch.setattr(evaluations._history, "close", fail_reader)
        exiting = asyncio.create_task(manager.__aexit__(
            type(primary) if primary is not None else None, primary, None,
        ))
        await asyncio.wait_for(entered.wait(), 10)
        exiting.cancel("context exit interrupted")
        await asyncio.sleep(0)
        assert not exiting.done()
        release.set()
        with pytest.raises(BaseExceptionGroup) as raised:
            await exiting

        def leaves(error: BaseException) -> list[BaseException]:
            if isinstance(error, BaseExceptionGroup):
                return [leaf for child in error.exceptions for leaf in leaves(child)]
            return [error]

        failures = leaves(raised.value)
        assert cleanup in failures
        assert (primary in failures) is body_failed
        cancellations = [error for error in failures if isinstance(error, asyncio.CancelledError)]
        assert len(cancellations) == 1
        assert str(cancellations[0]) == "context exit interrupted"
        assert persistence.ready
        monkeypatch.setattr(evaluations._history, "close", original_close)
        await evaluations.close()
