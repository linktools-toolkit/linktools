#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Callable graph captures retain the current invocation's accepted inputs."""

from collections.abc import Mapping

import pytest

from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy, EvaluationSpec,
    GraphTargetSpec, ScoreBundle, StartEvaluationRequest, TaskCaseInput,
)
from linktools.ai.runtime import CaptureGraphRequest, CaptureInputRequest, Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeInvocation, TaskNodeResultRef
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels, rule_scorer


async def _score(context: TaskNodeContext[None]) -> JsonValue:
    return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_mode", ("fixed_input", "reproject_input"))
async def test_callable_graph_recapture_uses_current_accepted_input(
    source_mode: str,
) -> None:
    preparation = "source"

    def normalize(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return {**value, "value": value["value"] + 10, "prepared": preparation}

    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    original = Task("capture.original", echo, effect_policy="none")
    current = Task("capture.current", echo, normalize=normalize, effect_policy="none")
    scorer = Task("capture.score", _score, effect_policy="none")
    async with Runtime.open("callable-graph", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.in_memory()) as runtime:
        engine = runtime.tasks.bind(original, current, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("target", task=original, input={"value": 1}),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "input"))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("source", 1), (
            CaseSpec.from_capture(CaseRef("source", "one", 1), capture=captured),
        )), principal=PRINCIPAL, idempotency_key="source-dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("current", task=current.ref),), (rule_scorer(scorer),), input_mode=source_mode,
            policy=EvaluationPolicy(allow_volatile=True)),
            PRINCIPAL, "source-evaluation"), engine=engine)
        assert (await run.wait(timeout_seconds=30)).result.completion == "complete"
        trial = (await run.trials()).items[0]
        graph = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
        accepted = await graph.result("target")
        current_execution = await graph.execution("target")
        captures = {}
        for graph_mode in ("declaration_graph", "materialized_graph"):
            request = CaptureGraphRequest(PRINCIPAL, "graph-" + graph_mode, mode=graph_mode)
            graph_capture = await runtime.tasks.capture_graph(graph.graph_id, request)
            assert await runtime.tasks.capture_graph(graph.graph_id, request) == graph_capture
            template = await runtime._input_captures.read_graph(graph_capture, principal=PRINCIPAL)
            contract = await runtime._input_captures.read_task(template.nodes[0].input_capture, principal=PRINCIPAL)
            assert contract.source_execution_id == current_execution.execution_id
            assert contract.task_ref == current.ref
            assert dict(contract.binding) == {"id": current.id, "revision": current.revision, **dict(current.contract)}
            assert dict(contract.input) == accepted
            assert dict(contract.original_input) == {"value": 1}
            assert contract.input_mode == "fixed_input"
            captures[graph_mode] = graph_capture

        for graph_mode, graph_capture in captures.items():
            dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef(graph_mode, 1), (
                CaseSpec.graph(CaseRef(graph_mode, "unchanged", 1), inputs={}),
                CaseSpec.graph(CaseRef(graph_mode, "addition", 1), inputs={"target": TaskCaseInput(input={"extra": True})}),
            )), principal=PRINCIPAL, idempotency_key="graph-dataset-" + graph_mode)
            for input_mode in ("fixed_input", "reproject_input"):
                preparation = input_mode
                replay = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                    (CandidateSpec("captured", graph_template=GraphTargetSpec(capture=graph_capture, outputs={"answer": "target"})),),
                    (rule_scorer(scorer),), input_mode=input_mode, policy=EvaluationPolicy(allow_volatile=True)),
                    PRINCIPAL, graph_mode + "-" + input_mode), engine=engine)
                assert (await replay.wait(timeout_seconds=30)).result.completion == "complete"
                for replay_trial in (await replay.trials()).items:
                    replay_graph = await engine.get(replay_trial.graph_ref.graph_id, principal=PRINCIPAL)
                    expected = accepted if input_mode == "fixed_input" else {"value": 11, "prepared": preparation}
                    if replay_trial.case_ref.case_id == "addition":
                        expected = {**expected, "extra": True}
                    assert await replay_graph.result("target") == expected


@pytest.mark.asyncio
async def test_callable_graph_recapture_keeps_live_and_frozen_dependencies() -> None:
    produced = 0

    async def produce(context: TaskNodeContext[None]) -> JsonValue:
        nonlocal produced
        produced += 1
        return produced

    async def consume(context: TaskNodeContext[None]) -> JsonValue:
        return {name: await context.read_dependency(name) for name in context.dependencies}

    producer = Task("capture.produce", produce, effect_policy="none")
    consumer = Task("capture.consume", consume, effect_policy="none")
    async with Runtime.open("callable-dependencies", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.in_memory()) as runtime:
        engine = runtime.tasks.bind(producer, consumer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("external", task=producer),
            TaskNode("target", ("external",), task=consumer, input_refs={"frozen": TaskNodeResultRef("external")}),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "input"))
        external = await source.result_ref("external")
        projected = await runtime._input_captures.task_input(captured, principal=PRINCIPAL, input_mode="reproject_input")
        graph = await engine.start(TaskGraph("current", (
            TaskNode("internal", task=producer),
            TaskNode("target", ("internal",), task=consumer, input_capture=projected,
                     input_refs={"live": TaskNodeResultRef("internal")}),
        )), principal=PRINCIPAL, idempotency_key="current")
        assert (await graph.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        capture = await runtime.tasks.capture_graph(graph.graph_id, CaptureGraphRequest(PRINCIPAL, "graph"))
        template = await runtime._input_captures.read_graph(capture, principal=PRINCIPAL)
        target = next(node for node in template.nodes if node.node_id == "target")
        contract = await runtime._input_captures.read_task(target.input_capture, principal=PRINCIPAL)
        assert contract.source_execution_id == (await graph.execution("target")).execution_id
        assert {item.name for item in contract.dependencies} == {"external", "frozen"}
        assert all(item.source_ref == external for item in contract.dependencies)
        assert target.input_refs == {"live": TaskNodeResultRef("internal")}
        replay = await engine.start(TaskGraph("replay", template.nodes), principal=PRINCIPAL, idempotency_key="replay")
        assert (await replay.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        assert await graph.result("target") == {"external": 1, "frozen": 1, "internal": 2, "live": 2}
        assert await replay.result("target") == {"external": 1, "frozen": 1, "internal": 3, "live": 3}


@pytest.mark.asyncio
async def test_callable_graph_capture_distinguishes_unstarted_from_missing_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def echo(context: TaskNodeContext[None]) -> JsonValue:
        return dict(context.input)

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.TASK_NODE_FAILED)

    async def unavailable(execution_id: str, invocation: TaskNodeInvocation) -> None:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    task = Task("capture.echo", echo, effect_policy="none")
    gate = Task("capture.gate", fail, effect_policy="none")
    async with Runtime.open("callable-unavailable", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.in_memory()) as runtime:
        engine = runtime.tasks.bind(task, gate)
        source = await engine.start(TaskGraph("source", (TaskNode("target", task=task, input={"value": 1}),)),
                                    principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=30)).result.wait_status is TaskStatus.SUCCEEDED
        execution = await source.execution("target")
        captured = await runtime.executions.capture_input(execution.execution_id, CaptureInputRequest(PRINCIPAL, "input"))
        for input_mode in ("fixed_input", "reproject_input"):
            projected = await runtime._input_captures.task_input(captured, principal=PRINCIPAL, input_mode=input_mode)
            blocked = await engine.start(TaskGraph(input_mode, (
                TaskNode("gate", task=gate),
                TaskNode("target", ("gate",), task=task, input_capture=projected),
            )), principal=PRINCIPAL, idempotency_key=input_mode)
            assert (await blocked.wait(timeout_seconds=30)).result.wait_status is TaskStatus.FAILED
            assert next(node.execution_id for node in (await blocked.state()).node_states if node.node_id == "target") is None
            for mode in ("declaration_graph", "materialized_graph"):
                request = CaptureGraphRequest(PRINCIPAL, input_mode + mode, mode=mode)
                if input_mode == "reproject_input":
                    with pytest.raises(AIError) as missing:
                        await runtime.tasks.capture_graph(blocked.graph_id, request)
                    assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
                    assert missing.value.safe_details["reason"] == "graph_node_never_started"
                else:
                    capture = await runtime.tasks.capture_graph(blocked.graph_id, request)
                    template = await runtime._input_captures.read_graph(capture, principal=PRINCIPAL)
                    target = next(node for node in template.nodes if node.node_id == "target")
                    assert target.input_capture == projected
                    contract = await runtime._input_captures.read_task(target.input_capture, principal=PRINCIPAL)
                    assert contract.source_execution_id == execution.execution_id
                    assert contract.input_mode == "fixed_input"

        monkeypatch.setattr(runtime._input_captures, "record_invocation", unavailable)
        interrupted = await engine.start(TaskGraph("interrupted", (
            TaskNode("target", task=task, input_capture=projected),
        )), principal=PRINCIPAL, idempotency_key="interrupted")
        assert (await interrupted.wait(timeout_seconds=30)).result.wait_status is TaskStatus.FAILED
        assert (await interrupted.execution("target")).execution_id != execution.execution_id
        for mode in ("declaration_graph", "materialized_graph"):
            with pytest.raises(AIError) as missing:
                await runtime.tasks.capture_graph(interrupted.graph_id, CaptureGraphRequest(PRINCIPAL, "missing-" + mode, mode=mode))
            assert missing.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
            assert missing.value.safe_details["reason"] == "task_invocation_not_retained"
