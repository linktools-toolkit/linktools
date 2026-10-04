#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Graph captures replay accepted attachments and the requested context policy."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import TaskStatus
from linktools.ai.runtime import (
    AgentTaskInput, CaptureGraphRequest, CaptureInputRequest, ExecutionInputContext,
    Runtime, RuntimeStorage,
)
from linktools.ai.task import TaskGraph, TaskNode
from linktools.ai.workspace import Workspace
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels


def _agents() -> CapabilityGroup:
    group = CapabilityGroup("capture-isolation")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize("file_change", ["unchanged", "changed", "deleted"])
@pytest.mark.parametrize("mode", ["declaration_graph", "materialized_graph"])
async def test_graph_capture_replays_each_accepted_attachment_once(
    tmp_path: Path, file_change: str, mode: str,
) -> None:
    work = tmp_path / "workspace"
    work.mkdir()
    attachment = work / "input.txt"
    attachment.write_text("accepted file bytes", encoding="utf-8")
    capabilities = (_agents(), CapabilityGroup("workspace", workspace=Workspace.load(work)))
    models = FixtureModels()
    async with Runtime.open("capture-files", models=models, context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path / "state"),
                            capabilities=capabilities) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question", files=("input.txt",))),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        assert models.attachments == [b"accepted file bytes"]
        capture = await runtime.tasks.capture_graph("source", CaptureGraphRequest(PRINCIPAL, "capture", mode=mode))
    if file_change == "changed":
        attachment.write_text("changed live file", encoding="utf-8")
    elif file_change == "deleted":
        attachment.unlink()
    async with Runtime.open("capture-files", models=models, context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path / "state"),
                            capabilities=capabilities) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        template = await runtime._input_captures.read_graph(capture, principal=PRINCIPAL)
        replay = await runtime.tasks.bind(task).start(TaskGraph("replay", template.nodes),
            principal=PRINCIPAL, idempotency_key="replay")
        assert (await replay.wait()).status is TaskStatus.SUCCEEDED
        assert models.attachments == [b"accepted file bytes", b"accepted file bytes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("input_form", ["inline", "reference"])
@pytest.mark.parametrize("context_policy", ["clean", "captured"])
async def test_graph_capture_applies_context_policy_to_all_agent_input_forms(
    tmp_path: Path, input_form: str, context_policy: str,
) -> None:
    context = ExecutionInputContext.from_messages((
        ModelRequest(parts=[UserPromptPart("OLD HISTORICAL QUESTION")]),
        ModelResponse(parts=[TextPart("OLD HISTORICAL ANSWER")]),
    ), session_metadata={"secret": "historical value"})
    async with Runtime.open("capture-context", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(_agents(),)) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        engine = runtime.tasks.bind(task)
        run = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question", input_context=context)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await run.wait()).status is TaskStatus.SUCCEEDED
        source_id = run.graph_id
        if input_form == "reference":
            execution = await run.execution("agent")
            agent_capture = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(PRINCIPAL, "agent-input"))
            task_capture = await runtime._input_captures.task_input(agent_capture, principal=PRINCIPAL)
            run = await engine.start(TaskGraph("reference", (
                TaskNode("agent", task=task, input_capture=task_capture),
            )), principal=PRINCIPAL, idempotency_key="reference")
            assert (await run.wait()).status is TaskStatus.SUCCEEDED
            source_id = run.graph_id
        capture = await runtime.tasks.capture_graph(source_id,
            CaptureGraphRequest(PRINCIPAL, "graph-capture", context_policy=context_policy))
        template = await runtime._input_captures.read_graph(capture, principal=PRINCIPAL)
        replay = await engine.start(TaskGraph("replay", template.nodes),
            principal=PRINCIPAL, idempotency_key="replay")
        assert (await replay.wait()).status is TaskStatus.SUCCEEDED
        execution = await replay.execution("agent")
        interactions = await execution.model_interactions(include_content=True)
        assert ("OLD HISTORICAL QUESTION" in str(interactions.items[0].request)) == (context_policy == "captured")
        assert "question" in str(interactions.items[0].request)
        recaptured = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(PRINCIPAL, "replayed-input"))
        value = await runtime._input_captures.read_agent(recaptured, principal=PRINCIPAL)
        assert dict(value.input_context.session_metadata) == (
            {"secret": "historical value"} if context_policy == "captured" else {}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["agent", "graph"])
@pytest.mark.parametrize("context_policy", ["clean", "captured"])
@pytest.mark.parametrize("input_mode", ["fixed_input", "reproject_input"])
async def test_capture_context_policy_remains_authoritative_when_reprojecting(
    tmp_path: Path, source_kind: str, context_policy: str, input_mode: str,
) -> None:
    from linktools.ai.evaluation import (
        CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
        EvaluationSpec, StartEvaluationRequest,
    )
    from linktools.ai.task import Task
    from .test_evaluation_consumers import exact, rule_scorer

    context = ExecutionInputContext.from_messages((
        ModelRequest(parts=[SystemPromptPart("PRIOR INSTRUCTIONS"), UserPromptPart("HISTORICAL INPUT TO REMOVE")]),
        ModelResponse(parts=[TextPart("prior answer")]),
    ), session_metadata={"secret": "prior metadata"})
    async with Runtime.open("capture-reproject", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(_agents(),)) as runtime:
        task = runtime.tasks.from_agent("capture.agent", runtime.agents.get())
        scorer = Task("capture.scorer", exact, effect_policy="none")
        engine = runtime.tasks.bind(task, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("current question", input_context=context)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait()).status is TaskStatus.SUCCEEDED
        if source_kind == "graph":
            graph_capture = await runtime.tasks.capture_graph(source.graph_id,
                CaptureGraphRequest(PRINCIPAL, "graph-capture", context_policy=context_policy))
            template = await runtime._input_captures.read_graph(graph_capture, principal=PRINCIPAL)
            source = await engine.start(TaskGraph("graph-replay", template.nodes),
                principal=PRINCIPAL, idempotency_key="graph-replay")
            assert (await source.wait()).status is TaskStatus.SUCCEEDED
        execution = await source.execution("agent")
        capture = await runtime.executions.capture_input(execution.execution_id,
            CaptureInputRequest(PRINCIPAL, "input-capture", context_policy=context_policy))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
            CaseSpec.from_capture(CaseRef("data", "case", 1), capture=capture, expected="fixture answer"),
        )), principal=PRINCIPAL, idempotency_key="publish")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("agent", task=task.ref),), (rule_scorer(scorer),), input_mode=input_mode,
            policy=EvaluationPolicy(model_fixtures=(FixtureModels.contract,))), PRINCIPAL, "evaluate"), engine=engine)
        assert (await run.wait(timeout_seconds=20)).completion == "complete"
        trial = (await run.trials()).items[0]
        interactions = await runtime.executions.model_interactions(trial.subject.execution_id,
            principal=PRINCIPAL, include_content=True)
        content = str(interactions.items[0].request)
        assert "current question" in content
        assert ("HISTORICAL INPUT TO REMOVE" in content) == (context_policy == "captured")
        assert ("PRIOR INSTRUCTIONS" in content) == (context_policy == "captured")
        recaptured = await runtime.executions.capture_input(trial.subject.execution_id,
            CaptureInputRequest(PRINCIPAL, "result-capture"))
        value = await runtime._input_captures.read_agent(recaptured, principal=PRINCIPAL)
        assert dict(value.input_context.session_metadata) == (
            {"secret": "prior metadata"} if context_policy == "captured" else {}
        )
