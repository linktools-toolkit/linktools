#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.evaluation_engine import Evaluations
from linktools.ai.task_engine import TaskEngine
from linktools.ai.task import Task

from linktools.ai.persistence import Persistence
"""Graph captures replay accepted attachments and the requested context policy."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import TaskStatus
from linktools.ai.inputs import AgentTaskInput, CaptureGraphRequest, CaptureInputRequest
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode
from linktools.ai.workspace import Workspace
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels


def _agents(*, workspace: Workspace | None = None) -> RuntimeFeatures:
    group = workspace_features("capture-isolation", workspace=workspace)
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
    features = _agents(workspace=Workspace.load(work))
    models = FixtureModels()
    async with Persistence.open('capture-files', storage=PersistenceStorage.filesystem(tmp_path / 'state'), tenant_id='evaluation-consumers') as persistence_42, Runtime.attach(persistence=persistence_42, context=CONTEXT, features=model_features(models, features)) as runtime, TaskEngine.open(persistence=persistence_42, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("capture.agent", runtime.agents.get())
        run = await runtime_engine.with_definitions(task).start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question", files=("input.txt",))),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert ((await run.wait()).result).status is TaskStatus.SUCCEEDED
        assert models.attachments == [b"accepted file bytes"]
        capture = await runtime_engine.inputs.capture_graph("source", CaptureGraphRequest(PRINCIPAL, "capture", mode=mode))
    if file_change == "changed":
        attachment.write_text("changed live file", encoding="utf-8")
    elif file_change == "deleted":
        attachment.unlink()
    async with Persistence.open('capture-files', storage=PersistenceStorage.filesystem(tmp_path / 'state'), tenant_id='evaluation-consumers') as persistence_56, Runtime.attach(persistence=persistence_56, context=CONTEXT, features=model_features(models, features)) as runtime, TaskEngine.open(persistence=persistence_56, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("capture.agent", runtime.agents.get())
        template = await runtime_engine.inputs.read_graph(capture, principal=PRINCIPAL)
        replay = await runtime_engine.with_definitions(task).start(TaskGraph("replay", template.nodes),
            principal=PRINCIPAL, idempotency_key="replay")
        assert ((await replay.wait()).result).status is TaskStatus.SUCCEEDED
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
    async with Persistence.open('capture-context', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_77, Runtime.attach(persistence=persistence_77, context=CONTEXT, features=model_features(FixtureModels(), _agents())) as runtime, TaskEngine.open(persistence=persistence_77, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("capture.agent", runtime.agents.get())
        engine = runtime_engine.with_definitions(task)
        run = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("question", input_context=context)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert ((await run.wait()).result).status is TaskStatus.SUCCEEDED
        source_id = run.graph_id
        if input_form == "reference":
            execution = await run.execution("agent")
            agent_capture = await runtime_engine.inputs.capture_input(execution.execution_id,
                CaptureInputRequest(PRINCIPAL, "agent-input"))
            task_capture = await runtime_engine.inputs.task_input(agent_capture, principal=PRINCIPAL)
            run = await engine.start(TaskGraph("reference", (
                TaskNode("agent", task=task, input_capture=task_capture),
            )), principal=PRINCIPAL, idempotency_key="reference")
            assert ((await run.wait()).result).status is TaskStatus.SUCCEEDED
            source_id = run.graph_id
        capture = await runtime_engine.inputs.capture_graph(source_id,
            CaptureGraphRequest(PRINCIPAL, "graph-capture", context_policy=context_policy))
        template = await runtime_engine.inputs.read_graph(capture, principal=PRINCIPAL)
        replay = await engine.start(TaskGraph("replay", template.nodes),
            principal=PRINCIPAL, idempotency_key="replay")
        assert ((await replay.wait()).result).status is TaskStatus.SUCCEEDED
        execution = await replay.execution("agent")
        interactions = await execution.model_interactions(include_content=True)
        assert ("OLD HISTORICAL QUESTION" in str(interactions.items[0].request)) == (context_policy == "captured")
        assert "question" in str(interactions.items[0].request)
        recaptured = await runtime_engine.inputs.capture_input(execution.execution_id,
            CaptureInputRequest(PRINCIPAL, "replayed-input"))
        value = await runtime_engine.inputs.read_agent(recaptured, principal=PRINCIPAL)
        assert dict(value.input_context.session_metadata) == (
            {"secret": "historical value"} if context_policy == "captured" else {}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("input_mode", "context_policy", "source_kind"), (
    ("fixed_input", "clean", "agent"),
    pytest.param("fixed_input", "clean", "graph", marks=pytest.mark.merge),
    pytest.param("fixed_input", "captured", "agent", marks=pytest.mark.merge),
    ("fixed_input", "captured", "graph"),
    ("reproject_input", "clean", "agent"),
    ("reproject_input", "clean", "graph"),
    ("reproject_input", "captured", "agent"),
    pytest.param("reproject_input", "captured", "graph", marks=pytest.mark.merge),
))
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
    async with Persistence.open('capture-reproject', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_139, Runtime.attach(persistence=persistence_139, context=CONTEXT, features=model_features(FixtureModels(), _agents())) as runtime, TaskEngine.open(persistence=persistence_139, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task = Task.from_agent("capture.agent", runtime.agents.get())
        scorer = Task("capture.scorer", exact, effect_policy="none")
        engine = runtime_engine.with_definitions(task, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=task, input=AgentTaskInput("current question", input_context=context)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert ((await source.wait()).result).status is TaskStatus.SUCCEEDED
        if source_kind == "graph":
            graph_capture = await runtime_engine.inputs.capture_graph(source.graph_id,
                CaptureGraphRequest(PRINCIPAL, "graph-capture", context_policy=context_policy))
            template = await runtime_engine.inputs.read_graph(graph_capture, principal=PRINCIPAL)
            source = await engine.start(TaskGraph("graph-replay", template.nodes),
                principal=PRINCIPAL, idempotency_key="graph-replay")
            assert ((await source.wait()).result).status is TaskStatus.SUCCEEDED
        execution = await source.execution("agent")
        capture = await runtime_engine.inputs.capture_input(execution.execution_id,
            CaptureInputRequest(PRINCIPAL, "input-capture", context_policy=context_policy))
        async with Evaluations.open(engine) as evaluations:
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), cases=(
                CaseSpec.from_capture(CaseRef("data", "case", 1), capture=capture, expected="fixture answer"),
            )), principal=PRINCIPAL, idempotency_key="publish")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("agent", task=task.ref),), (rule_scorer(scorer),), input_mode=input_mode,
                policy=EvaluationPolicy(model_fixtures=(FixtureModels.contract,))), PRINCIPAL, "evaluate"))
            assert ((await run.wait(timeout_seconds=20)).result).completion == "complete"
            trial = (await run.trials()).items[0]
            interactions = await runtime.executions.model_interactions(trial.subject.execution_id,
                principal=PRINCIPAL, include_content=True)
            content = str(interactions.items[0].request)
            assert "current question" in content
            assert ("HISTORICAL INPUT TO REMOVE" in content) == (context_policy == "captured")
            assert ("PRIOR INSTRUCTIONS" in content) == (context_policy == "captured")
            recaptured = await runtime_engine.inputs.capture_input(trial.subject.execution_id,
                CaptureInputRequest(PRINCIPAL, "result-capture"))
            value = await runtime_engine.inputs.read_agent(recaptured, principal=PRINCIPAL)
            assert dict(value.input_context.session_metadata) == (
                {"secret": "prior metadata"} if context_policy == "captured" else {}
            )
