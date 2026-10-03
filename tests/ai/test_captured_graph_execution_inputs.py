#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured graphs use accepted Agent inputs and only their executable bindings."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, ScoreBundle, ScoringInput, StartEvaluationRequest,
)
from linktools.ai.runtime import (
    AgentTaskInput, AgentTaskInputContext, CaptureGraphRequest, ExecutionInputContext,
    Runtime, RuntimeStorage,
)
from linktools.ai.task import Task, TaskExpander, TaskGraph, TaskNode, TaskNodeContext
from linktools.ai.workspace import Workspace
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels, rule_scorer


async def _score_graph(context: TaskNodeContext[None]) -> JsonValue:
    sample = ScoringInput.from_mapping(context.input)
    assert sample.target.status == "succeeded"
    return ScoreBundle(dimensions={"exact_match": 1.0}).to_mapping()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["declaration_graph", "materialized_graph"])
@pytest.mark.parametrize("context_policy", ["clean", "captured"])
@pytest.mark.parametrize("file_change", ["changed", "deleted"])
@pytest.mark.parametrize("input_mode", ["fixed_input", "reproject_input"])
async def test_projected_graph_capture_reuses_accepted_prompt_and_file_bytes(
    tmp_path: Path, mode: str, context_policy: str, file_change: str, input_mode: str,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    attachment = work / "file.txt"
    attachment.write_text("ORIGINAL", encoding="utf-8")
    group = CapabilityGroup("captured-projection")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    context = ExecutionInputContext.from_messages((
        ModelRequest(parts=[UserPromptPart("Historical question")]),
        ModelResponse(parts=[TextPart("Historical answer")]),
    ))
    models = FixtureModels()

    async def project(context: AgentTaskInputContext) -> str:
        return f"Prepared in {context.graph_id}: {context.input['question']}"

    async with Runtime.open("captured-projection", models=models, storage=RuntimeStorage.filesystem(tmp_path / "state"),
            context=CONTEXT, capabilities=(group, CapabilityGroup("work", workspace=Workspace.load(work)))) as runtime:
        target = runtime.tasks.from_agent("captured.agent", runtime.agents.get(), build_input=project)
        scorer = Task("captured.score", _score_graph, effect_policy="none")
        engine = runtime.tasks.bind(target, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=target, input=AgentTaskInput(parameters={"question": "Question"},
                files=("file.txt",), input_context=context)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
        capture = await runtime.tasks.capture_graph(source.graph_id,
            CaptureGraphRequest(PRINCIPAL, "capture", mode=mode, context_policy=context_policy))
        if file_change == "changed":
            attachment.write_text("MUTATED", encoding="utf-8")
        else:
            attachment.unlink()
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "case", 1), inputs={}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "agent"})),),
            (rule_scorer(scorer),), input_mode=input_mode,
            policy=EvaluationPolicy(model_fixtures=(models.contract,))),
            PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=20)).completion == "complete"
        report = await run.report()
        assert report.scores[0].valid == 1
        assert models.attachments == [b"ORIGINAL", b"ORIGINAL"]
        assert models.prompts[0].startswith("Prepared in source: Question")
        if input_mode == "fixed_input":
            assert models.prompts[1] == models.prompts[0]
        else:
            assert models.prompts[1].startswith("Prepared in ")
            assert models.prompts[1] != models.prompts[0]
            assert "Question" in models.prompts[1]
        trial = (await run.trials()).items[0]
        replay = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
        execution = await replay.execution("agent")
        interactions = await execution.model_interactions(include_content=True)
        assert ("Historical question" in str(interactions.items[0].request)) == (context_policy == "captured")


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "binding_scope"), [
    ("declaration_graph", "all"), ("materialized_graph", "all"), ("materialized_graph", "used"),
])
async def test_graph_capture_retains_only_bindings_required_by_its_execution_topology(
    tmp_path: Path, mode: str, binding_scope: str,
) -> None:
    expanded: list[str] = []

    async def seed(context: TaskNodeContext[None]) -> JsonValue:
        return "seed"

    async def child(context: TaskNodeContext[None]) -> JsonValue:
        return "child"

    seed_task = Task("captured.seed", seed, effect_policy="none")
    child_task = Task("captured.child", child, effect_policy="none")
    unused = Task("captured.unused", child, effect_policy="none")
    scorer = Task("captured.score", _score_graph, effect_policy="none")

    def expand(context: object) -> tuple[TaskNode, ...]:
        expanded.append("expanded")
        return (TaskNode("child", ("seed",), task=child_task),)

    expander = TaskExpander("captured.expand", expand)
    async with Runtime.open("captured-topology", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        engine = runtime.tasks.bind(seed_task, child_task, unused, expander)
        source = await engine.start(TaskGraph("source", (
            TaskNode("seed", task=seed_task, expander=expander),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
        capture = await runtime.tasks.capture_graph(source.graph_id,
            CaptureGraphRequest(PRINCIPAL, "capture", mode=mode))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "case", 1), inputs={}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        replay_engine = (runtime.tasks.bind(seed_task, child_task, scorer)
                         if binding_scope == "used"
                         else runtime.tasks.bind(seed_task, child_task, unused, scorer, expander))
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, selector="terminal_sinks")),),
            (rule_scorer(scorer),)), PRINCIPAL, "evaluate"), engine=replay_engine)
        assert (await run.wait(timeout_seconds=20)).completion == "complete"
        report = await run.report()
        assert report.scores[0].valid == 1
        assert len(expanded) == (1 if mode == "materialized_graph" else 2)
        trial = (await run.trials()).items[0]
        evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
        assert evidence.target.node_statuses == {"seed": "succeeded", "child": "succeeded"}


@pytest.mark.asyncio
@pytest.mark.parametrize("context_policy", ["clean", "captured"])
async def test_graph_capture_rejects_unavailable_projected_input(
    tmp_path: Path, context_policy: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    group = CapabilityGroup("unavailable-projection")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    models = FixtureModels()

    async def fail(context: TaskNodeContext[None]) -> JsonValue:
        raise AIError(ErrorCode.TASK_NODE_FAILED)

    async def project(context: AgentTaskInputContext) -> str:
        return "should never be evaluated"

    async with Runtime.open("unavailable-projection", models=models, context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
        failure = Task("captured.failure", fail, effect_policy="none")
        target = runtime.tasks.from_agent("captured.agent", runtime.agents.get(), build_input=project)
        source = await runtime.tasks.bind(failure, target).start(TaskGraph("source", (
            TaskNode("failure", task=failure),
            TaskNode("agent", ("failure",), task=target, input=AgentTaskInput(parameters={"question": "unrun"})),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.FAILED
        with pytest.raises(AIError) as raised:
            await runtime.tasks.capture_graph(source.graph_id,
                CaptureGraphRequest(PRINCIPAL, "capture", context_policy=context_policy))
        assert raised.value.code is (ErrorCode.INPUT_CONTEXT_UNAVAILABLE if context_policy == "captured"
                                     else ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
        assert raised.value.safe_details["reason"] == "graph_node_never_started"
        assert not models.prompts


@pytest.mark.asyncio
async def test_fixed_agent_task_capture_can_reproject_without_live_files(tmp_path: Path) -> None:
    from linktools.ai.runtime import CaptureInputRequest

    work = tmp_path / "work"
    work.mkdir()
    attachment = work / "file.txt"
    attachment.write_text("ORIGINAL", encoding="utf-8")
    models = FixtureModels()
    group = CapabilityGroup("recaptured-projection")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def project(context: AgentTaskInputContext) -> str:
        return f"Prepared in {context.graph_id}: {context.input['question']}"

    async with Runtime.open("recaptured-projection", models=models, context=CONTEXT,
            storage=RuntimeStorage.filesystem(tmp_path / "state"),
            capabilities=(group, CapabilityGroup("work", workspace=Workspace.load(work)))) as runtime:
        task = runtime.tasks.from_agent("captured.agent", runtime.agents.get(), build_input=project)
        scorer = Task("captured.score", _score_graph, effect_policy="none")
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput(parameters={"question": "Question"}, files=("file.txt",))),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
        execution = await source.execution("agent")
        agent_capture = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(PRINCIPAL, "agent-input", context_policy="clean"))
        task_capture = await runtime._input_captures.task_input(agent_capture, principal=PRINCIPAL)
        attachment.unlink()
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.from_capture(CaseRef("data", "case", 1), capture=task_capture),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("captured", task=task.ref),), (rule_scorer(scorer),), input_mode="reproject_input",
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=20)).completion == "complete"
        assert (await run.report()).scores[0].valid == 1
        assert models.attachments == [b"ORIGINAL", b"ORIGINAL"]
        assert models.prompts[1] != models.prompts[0]


@pytest.mark.asyncio
async def test_clean_unstarted_literal_graph_can_reproject_its_admitted_files(tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.evaluation import TaskCaseInput

    work = tmp_path / "work"
    work.mkdir()
    attachment = work / "file.txt"
    attachment.write_text("ORIGINAL", encoding="utf-8")
    models = FixtureModels()
    group = CapabilityGroup("unstarted-literal")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def prepare(context: TaskNodeContext[None]) -> JsonValue:
        if not context.input.get("succeed"):
            raise AIError(ErrorCode.TASK_NODE_FAILED)
        return "ready"

    async with Runtime.open("unstarted-literal", models=models, context=CONTEXT,
            storage=RuntimeStorage.filesystem(tmp_path / "state"),
            capabilities=(group, CapabilityGroup("work", workspace=Workspace.load(work)))) as runtime:
        producer = Task("captured.prepare", prepare, effect_policy="none")
        target = runtime.tasks.from_agent("captured.agent", runtime.agents.get())
        scorer = Task("captured.score", _score_graph, effect_policy="none")
        engine = runtime.tasks.bind(producer, target, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("prepare", task=producer),
            TaskNode("agent", ("prepare",), task=target, input=AgentTaskInput("Question", files=("file.txt",))),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.FAILED
        assert not models.prompts
        capture = await runtime.tasks.capture_graph(source.graph_id,
            CaptureGraphRequest(PRINCIPAL, "capture", context_policy="clean"))
        attachment.unlink()
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
            CaseSpec.graph(CaseRef("data", "case", 1), inputs={"prepare": TaskCaseInput(input={"succeed": True})}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "agent"})),),
            (rule_scorer(scorer),), input_mode="reproject_input",
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=20)).completion == "complete"
        assert (await run.report()).scores[0].valid == 1
        assert models.attachments == [b"ORIGINAL"]
        assert models.prompts[0].startswith("Question")
