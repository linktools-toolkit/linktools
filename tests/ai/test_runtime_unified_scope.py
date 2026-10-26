#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified composition retains independent execution and resource ownership."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import CheckpointCapability
from linktools.ai.core import (
    AuthorizationAction, ExecutionContext, ExecutionStatus, JsonValue, Principal,
    ResourceRef, TaskStatus, service_principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, DimensionContract,
    EvaluationSpec, ScorerSpec, ScoringInput, ScoreBundle, StartEvaluationRequest,
)
from linktools.ai.evaluation_engine import Evaluations
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.observe import Metrics
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
from linktools.ai.task_engine import TaskEngine


class _OfflineBinding:
    provider = "test"
    model_identity = "test:scope"
    contract = {"provider": provider, "model_identity": model_identity}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="scope answer")


def _features() -> RuntimeFeatures:
    features = RuntimeFeatures()
    features.model("default", _OfflineBinding())
    return features


async def _echo(context: TaskNodeContext[object]) -> JsonValue:
    return context.input.get("answer", "echo")


def _graph(task: Task, graph_id: str = "graph") -> TaskGraph:
    return TaskGraph(graph_id, (TaskNode("echo", task=task, input={"answer": "yes"}),))


@pytest.mark.asyncio
async def test_task_only_scope_does_not_compose_an_agent_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime import _factory

    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("task-only scope activated an Agent producer")

    monkeypatch.setattr(_factory, "compose_runtime_components", forbidden)
    monkeypatch.setattr(Persistence, "create_agent_run_store", forbidden)
    task = Task("echo", _echo, effect_policy="none")
    async with Runtime.open("task-only", storage=PersistenceStorage.in_memory()) as runtime:
        assert runtime.metrics.status().recorders == ()
        engine = await runtime.tasks.bind(task)
        persistence = engine.persistence
        run = await engine.start(_graph(task), idempotency_key="start")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert await run.result("echo") == "yes"
        for name in ("sessions", "executions", "approvals", "external_calls", "events", "artifacts", "history"):
            with pytest.raises(AIError) as error:
                getattr(runtime, name)
            assert error.value.safe_details["reason"] == "agent_domain_not_configured"
        with pytest.raises(AIError):
            runtime.agents.get()
    assert not persistence.ready
    with pytest.raises(AIError):
        await engine.start(_graph(task, "after-close"), idempotency_key="after-close")


@pytest.mark.asyncio
async def test_agent_and_agent_task_share_the_runtime_scope() -> None:
    async with Runtime.open("agent-task", storage=PersistenceStorage.in_memory(),
                            features=_features(), metrics=Metrics.in_memory()) as runtime:
        agent = runtime.agents.get()
        assert agent.runtime is runtime
        assert (await agent.run("answer", timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        task = Task.from_agent("agent.task", agent)
        engine = await runtime.tasks.bind(task)
        graph = TaskGraph("agent-graph", (TaskNode("answer", task=task, input=AgentTaskInput("answer")),))
        run = await engine.start(graph, idempotency_key="start")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert await run.result("answer") == {"text": "scope answer"}
        flushed = await runtime.metrics.flush()
        assert flushed.completed and len(flushed.status.recorders) == 2


@pytest.mark.asyncio
async def test_attached_scope_does_not_close_borrowed_persistence() -> None:
    task = Task("echo", _echo, effect_policy="none")
    async with Persistence.open("shared", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence) as runtime:
            engine = await runtime.tasks.bind(task)
            assert engine.persistence is persistence
        assert persistence.ready
        async with TaskEngine.open(task, persistence=persistence) as peer:
            run = await peer.start(_graph(task), idempotency_key="peer")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_scope_context_and_metrics_are_shared_without_duplicate_view_recorders() -> None:
    metrics = Metrics.in_memory()
    context = ExecutionContext({"value": 7}, metric_dimensions={"component": "scope"})

    async def read_app(ctx: TaskNodeContext[dict]) -> JsonValue:
        return ctx.app["value"]

    task = Task("read-app", read_app, effect_policy="none")
    async with Runtime.open("metrics", storage=PersistenceStorage.in_memory(), context=context, metrics=metrics) as runtime:
        first = await runtime.tasks.bind(task)
        second = await runtime.tasks.bind(task)
        assert first.context is second.context is context
        assert not first.shares_execution_owner(second)
        extra = Task("extra", _echo, effect_policy="none")
        view = first.with_definitions(extra)
        assert view.shares_execution_owner(first)
        run = await view.start(_graph(task), idempotency_key="start")
        await run.wait(timeout_seconds=5)
        assert await run.result("echo") == 7
        result = await runtime.metrics.flush()
        assert result.completed
        assert len(result.status.recorders) == 2
        assert result.status.accepted == sum(item.accepted for item in result.status.recorders)
        await view.close()
        assert len(runtime.metrics.status().recorders) == 2
        another = await first.start(_graph(task, "another"), idempotency_key="another")
        await another.wait(timeout_seconds=5)


class _Policy:
    def __init__(self, denied: AuthorizationAction | None = None) -> None:
        self.denied = denied
        self.actions: list[AuthorizationAction] = []

    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        self.actions.append(action)
        if action is self.denied:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


@pytest.mark.asyncio
async def test_top_level_policy_checks_task_and_agent_admission_independently() -> None:
    policy = _Policy()
    async with Runtime.open("policy", storage=PersistenceStorage.in_memory(), features=_features(), authorization=policy) as runtime:
        task = Task.from_agent("agent.task", runtime.agents.get())
        engine = await runtime.tasks.bind(task)
        assert engine.authorization is policy
        graph = TaskGraph("policy-graph", (TaskNode("answer", task=task, input=AgentTaskInput("answer")),))
        run = await engine.start(graph, idempotency_key="run")
        await run.wait(timeout_seconds=5)
        assert AuthorizationAction.TASK_RUN in policy.actions
        assert AuthorizationAction.EXECUTION_RUN in policy.actions
        policy.denied = AuthorizationAction.TASK_RUN
        with pytest.raises(AIError):
            await engine.start(TaskGraph("denied", graph.nodes), idempotency_key="denied")
        policy.denied = AuthorizationAction.EXECUTION_RUN
        with pytest.raises(AIError):
            await runtime.agents.get().start("denied")



@pytest.mark.asyncio
async def test_second_enter_rejection_does_not_close_the_active_scope() -> None:
    runtime = Runtime.open("one-enter", storage=PersistenceStorage.in_memory())
    async with runtime:
        with pytest.raises(AIError):
            await runtime.__aenter__()
        engine = await runtime.tasks.bind()
        assert engine.persistence.ready
    with pytest.raises(AIError):
        await runtime.__aenter__()


@pytest.mark.asyncio
async def test_close_failure_retains_persistence_and_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("close-retry", storage=PersistenceStorage.in_memory())
    await runtime.__aenter__()
    engine = await runtime.tasks.bind()
    persistence = engine.persistence
    original = engine.close
    attempts = 0

    async def close() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await original()

    monkeypatch.setattr(engine, "close", close)
    with pytest.raises(AIError):
        await runtime.close()
    assert persistence.ready
    with pytest.raises(AIError):
        await runtime.tasks.bind()
    await runtime.close()
    assert attempts == 2 and not persistence.ready
    await runtime.close()
    assert attempts == 2


@pytest.mark.asyncio
async def test_cancelled_close_keeps_cleanup_owned_until_it_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("cancel-close", storage=PersistenceStorage.in_memory())
    await runtime.__aenter__()
    engine = await runtime.tasks.bind()
    entered, release = asyncio.Event(), asyncio.Event()
    original = engine.close

    async def close() -> None:
        entered.set()
        await release.wait()
        await original()

    monkeypatch.setattr(engine, "close", close)
    closing = asyncio.create_task(runtime.close())
    await entered.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert engine.persistence.ready
    release.set()
    await runtime.close()
    assert not engine.persistence.ready


@pytest.mark.asyncio
async def test_body_error_is_preserved_when_cleanup_also_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Runtime.open("body-failure", storage=PersistenceStorage.in_memory())
    primary = ValueError("body failed")
    fail_cleanup = True
    with pytest.raises(ValueError) as caught:
        async with runtime:
            engine = await runtime.tasks.bind()
            original = engine.close

            async def fail() -> None:
                if fail_cleanup:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                await original()

            monkeypatch.setattr(engine, "close", fail)
            raise primary
    assert caught.value is primary and isinstance(caught.value.__cause__, AIError)
    assert engine.persistence.ready
    fail_cleanup = False
    await runtime.close()
    assert not engine.persistence.ready


@pytest.mark.asyncio
async def test_partial_agent_initialization_retains_cleanup_for_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime import _factory

    initialized = []
    initialize = Persistence.initialize
    compose = _factory.compose_runtime_components
    cleanup = _factory._AgentLifecycle.close
    fail_cleanup = True
    primary = RuntimeError("composition rejected after resource acquisition")

    async def capture(owner: Persistence, **kwargs: object) -> None:
        initialized.append(owner)
        await initialize(owner, **kwargs)

    async def fail_after_composition(**kwargs: object) -> object:
        await compose(**kwargs)
        raise primary

    async def close(owner: object) -> None:
        if fail_cleanup:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await cleanup(owner)

    monkeypatch.setattr(Persistence, "initialize", capture)
    monkeypatch.setattr(_factory, "compose_runtime_components", fail_after_composition)
    monkeypatch.setattr(_factory._AgentLifecycle, "close", close)
    runtime = Runtime.open("startup-failure", storage=PersistenceStorage.in_memory(), features=_features())
    with pytest.raises(RuntimeError) as caught:
        await runtime.__aenter__()
    assert caught.value is primary and isinstance(primary.__cause__, AIError)
    assert initialized[0].ready
    fail_cleanup = False
    await runtime.close()
    assert not initialized[0].ready


@pytest.mark.asyncio
async def test_single_scope_evaluation_uses_the_bound_task_engine(tmp_path: Path) -> None:
    async def exact(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        return ScoreBundle(dimensions={"exact": float(sample.target_output == sample.expected)}).to_mapping()

    target = Task("echo", _echo, effect_policy="none")
    judge = Task("exact", exact, effect_policy="none")
    dimension = DimensionContract("exact", "boolean", "higher", 0, 1)
    principal = service_principal("default", "evaluator")
    async with Runtime.open("evaluation", storage=PersistenceStorage.filesystem(tmp_path)) as runtime:
        engine = await runtime.tasks.bind(target, judge)
        evaluations = await runtime.evaluations.bind(engine)
        dataset = await evaluations.publish_dataset(
            DatasetSpec(DatasetRef("examples", 1), cases=(
                CaseSpec.task(CaseRef("examples", "yes", 1), input={"answer": "yes"}, expected="yes"),
            )), principal=principal, idempotency_key="dataset",
        )
        request = StartEvaluationRequest(
            EvaluationSpec(dataset, (CandidateSpec("echo", task=target.ref),),
                           (ScorerSpec("exact", judge.ref, (dimension,)),)),
            principal, "evaluation",
        )
        run = await evaluations.start(request)
        assert (await run.wait(timeout_seconds=10)).result.completion == "complete"
        assert (await run.create_report()).scores[0].mean == 1.0


@pytest.mark.asyncio
async def test_bound_evaluation_borrows_external_engine_without_closing_its_scope() -> None:
    task = Task("echo", _echo, effect_policy="none")
    async with Persistence.open("external", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(task, persistence=persistence) as external:
            async with Runtime.open("host", storage=PersistenceStorage.in_memory()) as runtime:
                await runtime.evaluations.bind(external)
                assert runtime.metrics.status().recorders == ()
            assert persistence.ready
            run = await external.start(_graph(task), idempotency_key="external")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_evaluations_stop_before_tasks_and_failed_task_close_does_not_close_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    runtime = Runtime.open("close-order", storage=PersistenceStorage.in_memory())
    await runtime.__aenter__()
    engine = await runtime.tasks.bind()
    original_close = engine.close
    fail = True

    class Evaluation:
        def preflight_handoff(self) -> None:
            return None

        def stop_admission(self) -> None:
            events.append("stop-evaluation")

        async def close(self) -> None:
            events.append("evaluation")

    @asynccontextmanager
    async def opened(*args: object, **kwargs: object):
        yield Evaluation()
        events.append("evaluation-exit")

    async def close_engine() -> None:
        events.append("task")
        if fail:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        await original_close()

    monkeypatch.setattr(Evaluations, "open", opened)
    monkeypatch.setattr(engine, "close", close_engine)
    await runtime.evaluations.bind(engine)
    with pytest.raises(AIError):
        await runtime.close()
    assert events == ["stop-evaluation", "evaluation", "evaluation-exit", "task"]
    assert engine.persistence.ready
    fail = False
    await runtime.close()
    assert events[-1] == "task" and events.count("evaluation") == 1
    assert not engine.persistence.ready


@pytest.mark.asyncio
async def test_explicit_external_agent_keeps_its_owner_and_authorization() -> None:
    agent_policy, task_policy = _Policy(), _Policy()
    async with Persistence.open("shared-agent", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(), authorization=agent_policy) as agent_owner:
            async with Runtime.attach(persistence=persistence, authorization=task_policy) as task_owner:
                task = Task.from_agent("external.agent", agent_owner.agents.get())
                engine = await task_owner.tasks.bind(task)
                graph = TaskGraph("external", (TaskNode("answer", task=task, input=AgentTaskInput("answer")),))
                run = await engine.start(graph, idempotency_key="external")
                assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
                assert AuthorizationAction.TASK_RUN in task_policy.actions
                assert AuthorizationAction.EXECUTION_RUN in agent_policy.actions
                assert AuthorizationAction.EXECUTION_RUN not in task_policy.actions
            assert persistence.ready
            assert (await agent_owner.agents.get().run("still open", timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_handoff_supports_idle_task_and_evaluation_domains(tmp_path: Path) -> None:
    features = _features()
    features.capability(CheckpointCapability())
    runtime = Runtime.open("handoff", storage=PersistenceStorage.filesystem(tmp_path), features=features)
    async with runtime:
        engine = await runtime.tasks.bind()
        await runtime.evaluations.bind(engine)
        persistence = engine.persistence
        await runtime.handoff()
        assert not persistence.ready


@pytest.mark.asyncio
async def test_handoff_rejects_inflight_evaluation_without_closing_other_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _features()
    features.capability(CheckpointCapability())
    runtime = Runtime.open("handoff-admission", storage=PersistenceStorage.filesystem(tmp_path), features=features)
    entered, release = asyncio.Event(), asyncio.Event()
    async with runtime:
        task = Task("echo", _echo, effect_policy="none")
        engine = await runtime.tasks.bind(task)
        evaluations = await runtime.evaluations.bind(engine)
        request = StartEvaluationRequest(
            EvaluationSpec(DatasetRef("unused", 1), (CandidateSpec("echo", task=task.ref),),
                           (ScorerSpec("judge", task.ref, (DimensionContract("exact", "boolean", "higher", 0, 1),)),)),
            service_principal("default", "evaluator"), "blocked",
        )

        async def blocked(*args: object, **kwargs: object) -> None:
            entered.set()
            await release.wait()
            raise ValueError("admission not committed")

        monkeypatch.setattr(evaluations, "_reserve_start", blocked)
        admission = asyncio.create_task(evaluations.start(request))
        await entered.wait()
        with pytest.raises(AIError) as error:
            await runtime.handoff()
        assert error.value.safe_details["reason"] == "active_evaluation"
        assert engine.persistence.ready
        # Rejected handoff must not freeze the still-open Runtime or its peers.
        peer = await runtime.tasks.bind()
        assert peer.persistence is engine.persistence
        release.set()
        with pytest.raises(ValueError):
            await admission
        await runtime.handoff()
        assert not engine.persistence.ready
