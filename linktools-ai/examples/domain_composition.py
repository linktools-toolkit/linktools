#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline examples of independent domains and Runtime-owned composition.

Uses only public APIs. Temporary local SQLite state is explicitly provisioned by
PersistenceStorage.from_root; it is not a deployment database migration example.
Each function is a complete scenario, with its resource lifetime visible.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai.models.test import TestModel

from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.core import JsonValue, TaskStatus, service_principal
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec,
    DimensionContract, EvaluationPolicy, EvaluationSpec, ScoreBundle,
    ScorerSpec, ScoringInput, StartEvaluationRequest,
)
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricQuery, Metrics, MetricSource,
    MetricType, MetricWindow,
)
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeResultRef
from linktools.ai.task.engine import TaskEngine


class OfflineModel:
    """A host-supplied ModelBinding backed by the SDK's offline test model."""

    provider = "test"
    model_identity = "test:domain-composition"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "domain-composition"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="answer")


async def echo(context: TaskNodeContext[None]) -> JsonValue:
    return context.input["answer"]


async def exact(context: TaskNodeContext[None]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    return ScoreBundle(dimensions={
        "exact_match": float(sample.expected_present and sample.target_output == sample.expected),
    }).to_mapping()


async def read_answer(context: TaskNodeContext[None]) -> JsonValue:
    return await context.read_dependency("answer")


async def task_only_evaluation(root: Path) -> None:
    """No Agent, model binding, or Runtime is constructed in this scenario."""
    target = Task("demo.echo", echo, effect_policy="none")
    scorer = Task("demo.exact", exact, effect_policy="none")
    principal = service_principal("demo", "evaluation-owner")
    metrics = Metrics.in_memory()
    async with (
        Persistence.open("task-example", storage=PersistenceStorage.from_root(root, exclusive_writer=True),
                         tenant_id="demo") as persistence,
        TaskEngine.open(target, scorer, persistence=persistence, metrics=metrics) as engine,
    ):
        graph = await engine.start(TaskGraph("echo", (
            TaskNode("echo", task=target, input={"answer": "hello"}),
        )), principal=principal, idempotency_key="echo")
        assert (await graph.wait(timeout_seconds=30)).result.status is TaskStatus.SUCCEEDED
        assert await graph.result("echo") == "hello"
        async with Evaluations.open(engine) as evaluations:
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("echo", 1), cases=(
                CaseSpec.task(CaseRef("echo", "hello", 1), input={"answer": "hello"}, expected="hello"),
            )), principal=principal, idempotency_key="publish")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(
                dataset, (CandidateSpec("current", task=target.ref),),
                (ScorerSpec("exact", task=scorer.ref,
                            dimensions=(DimensionContract("exact_match", "boolean", "higher", 0, 1),)),),
            ), principal, "evaluate"))
            view = (await run.wait(timeout_seconds=30)).result
            assert view.completion == "complete", view.needs_attention
            assert (await run.create_report()).scores[0].mean == 1.0
        # Evaluation closes its coordination, not this borrowed Task producer.
        assert (await graph.state()).status is TaskStatus.SUCCEEDED
    print("Task-only graph and evaluation passed")


async def shared_agent_and_tasks(root: Path) -> None:
    """Task borrows the existing Agent producer and the same Persistence."""
    features = RuntimeFeatures("shared-agent")
    features.model("offline", OfflineModel())
    features.agent("default", model="offline", allow_tools=())
    metrics = Metrics.in_memory()
    async with Persistence.open("shared-example", storage=PersistenceStorage.from_root(root, exclusive_writer=True)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features, metrics=metrics) as runtime:
            agent = runtime.agents.get()
            assert (await (await agent.start("question")).wait()).result.output == {"text": "answer"}
            target = Task.from_agent("demo.agent", agent)
            consumer = Task("demo.consume", read_answer, effect_policy="none")
            async with TaskEngine.open(target, consumer, persistence=persistence, metrics=metrics) as engine:
                graph = await engine.start(TaskGraph("mixed", (
                    TaskNode("agent", task=target, input=AgentTaskInput("question")),
                    TaskNode("consumer", task=consumer, dependencies=("agent",),
                             input_refs={"answer": TaskNodeResultRef("agent")}),
                )), idempotency_key="mixed")
                assert (await graph.wait(timeout_seconds=30)).result.status is TaskStatus.SUCCEEDED
                assert await graph.result("consumer") == {"text": "answer"}
            # Closing the Task borrower does not close its Agent producer.
            assert (await (await agent.start("another question")).wait()).result.output == {"text": "answer"}
        assert persistence.ready
    print("Shared Agent and mixed Task graph passed")


async def composed_agent_evaluation(root: Path) -> None:
    """Runtime owns all composed domains; callers need only one context."""
    model = OfflineModel()
    features = RuntimeFeatures("composed-agent")
    features.capability(CheckpointCapability(), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.model("offline", model)
    features.agent("default", model="offline", allow_tools=())
    principal = service_principal("demo", "evaluation-owner")
    async with Runtime.open("composed-example", storage=PersistenceStorage.from_root(root, exclusive_writer=True),
                            tenant_id="demo", features=features) as runtime:
        target = Task.from_agent("demo.agent", runtime.agents.get())
        scorer = Task("demo.exact", exact, effect_policy="none")
        engine = await runtime.tasks.bind(target, scorer)
        evaluations = await runtime.evaluations.bind(engine)
        dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("agent", 1), cases=(
            CaseSpec.agent(CaseRef("agent", "answer", 1), prompt="question", expected={"text": "answer"}),
        )), principal=principal, idempotency_key="publish")
        run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(
            dataset, (CandidateSpec("current", task=target.ref),),
            (ScorerSpec("exact", task=scorer.ref,
                        dimensions=(DimensionContract("exact_match", "boolean", "higher", 0, 1),)),),
            policy=EvaluationPolicy(model_fixtures=(model.contract,)),
        ), principal, "evaluate"))
        view = (await run.wait(timeout_seconds=30)).result
        assert view.completion == "complete", view.needs_attention
        assert (await run.create_report()).scores[0].mean == 1.0
    print("One-context Agent evaluation passed")


async def standalone_metrics() -> None:
    """Explicit measurement needs no Runtime, Agent, Task, or recorder lifetime."""
    metrics = Metrics.in_memory()
    await metrics.define(MetricDefinition(
        name="demo.items", revision=1, observation_kind="demo.sample",
        source=MetricSource.measurement("value"), metric_type=MetricType.COUNTER,
        unit="1", default_aggregation=MetricAggregation.SUM,
    ))
    await metrics.record("demo.items", 3, observation_id="batch-1")
    result = await metrics.query(MetricQuery("demo.items", MetricWindow.recent(minutes=1)))
    assert result.points[0].value == 3
    print("Standalone Metrics passed")


async def main() -> None:
    await standalone_metrics()
    with TemporaryDirectory(prefix="linktools-domains-") as directory:
        root = Path(directory)
        await task_only_evaluation(root / "tasks")
        await shared_agent_and_tasks(root / "shared")
        await composed_agent_evaluation(root / "composed")


if __name__ == "__main__":
    asyncio.run(main())
