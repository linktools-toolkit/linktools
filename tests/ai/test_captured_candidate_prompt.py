#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.evaluation_engine import Evaluations
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Captured conversation input uses the selected candidate's Agent behavior."""
from ._runtime_test_helpers import model_features

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, StartEvaluationRequest,
)
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskGraphTemplate, TaskNode

from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, FixtureModels,
    exact, rule_scorer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind,input_mode,target_kind", (
    ("fork", "fixed_input", "task"),
    ("task", "fixed_input", "task"),
    ("task", "reproject_input", "task"),
    ("fork", "fixed_input", "graph"),
    ("task-reference", "reproject_input", "graph"),
    ("task-reference", "fixed_input", "captured-graph"),
))
async def test_capture_uses_selected_candidate_prompt_and_preserves_history(
    source_kind: str, input_mode: str, target_kind: str,
) -> None:
    models = FixtureModels()
    group = RuntimeFeatures("fork-candidate")
    group.capability(TraceCapability(), scope="runtime")
    if source_kind == "fork":
        group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    original_prompt = "Original Agent behavior."
    candidate_prompt = "Selected candidate behavior."
    for name, prompt in (("default", original_prompt), ("candidate", candidate_prompt)):
        group.agent(name, model="default", system_prompt=prompt,
                    allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('fork-candidate', storage=PersistenceStorage.in_memory(), tenant_id='evaluation-consumers') as persistence_47, Runtime.attach(persistence=persistence_47, context=CONTEXT, features=model_features(models, group)) as runtime, TaskEngine.open(persistence=persistence_47, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        if source_kind == "fork":
            first = await runtime.agents.get().start("historical question", principal=PRINCIPAL)
            assert (await first.wait()).result.status is ExecutionStatus.SUCCEEDED
            source = await first.fork("captured question")
            assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        else:
            context = ExecutionInputContext.from_messages((
                ModelRequest(parts=[SystemPromptPart(original_prompt), UserPromptPart("historical question")]),
                ModelResponse(parts=[TextPart("fixture answer")]),
            ))
            source_task = Task.from_agent("candidate.source", runtime.agents.get())
            graph = await runtime_engine.with_definitions(source_task).start(TaskGraph("source", (
                TaskNode("agent", task=source_task, input=AgentTaskInput("captured question", input_context=context)),
            )), principal=PRINCIPAL, idempotency_key="source")
            assert (await graph.wait()).result.wait_status is TaskStatus.SUCCEEDED
            source = await graph.execution("agent")
        native_request = str((await source.model_interactions(include_content=True)).items[0].request)
        assert native_request.count(original_prompt) == 1
        assert candidate_prompt not in native_request
        capture = await runtime_engine.inputs.capture_input(source.execution_id, CaptureInputRequest(PRINCIPAL, "capture-fork"))
        original_capture = await runtime_engine.inputs.read_agent(capture, principal=PRINCIPAL)
        original_capture.require_importable()
        original = Task.from_agent("candidate.original", runtime.agents.restore(original_capture.binding))
        selected = Task.from_agent("candidate.selected", runtime.agents.get("candidate"))
        scorer = Task("candidate.score", exact, effect_policy="none")
        engine = runtime_engine.with_definitions(original, selected, scorer)
        case_capture = (await runtime_engine.inputs.task_input(capture, principal=PRINCIPAL)
                        if source_kind == "task-reference" else capture)
        case = CaseSpec.from_capture(CaseRef("fork-input", "one", 1), capture=case_capture, expected="fixture answer")
        if target_kind != "task":
            case = CaseSpec.graph(case.ref, inputs={} if target_kind == "captured-graph" else {"target": case.input},
                expected={"answer": {"status": "succeeded", "value": "fixture answer", "reason": None}})
            candidates = tuple(CandidateSpec(slot, graph_template=GraphTargetSpec(
                template=TaskGraphTemplate((TaskNode("target", task=task,
                    input_capture=case_capture if target_kind == "captured-graph" else None),)), outputs={"answer": "target"}))
                for slot, task in (("original", original), ("selected", selected)))
        else:
            candidates = (CandidateSpec("original", task=original.ref), CandidateSpec("selected", task=selected.ref))
        async with Evaluations.open(engine) as evaluations:
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("fork-input", 1), (
                case,
            )), principal=PRINCIPAL, idempotency_key="publish-fork")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                candidates,
                (rule_scorer(scorer),), input_mode=input_mode, policy=EvaluationPolicy(allow_volatile=True, model_fixtures=(models.contract,))),
                PRINCIPAL, "evaluate-fork"))
            view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
            assert view.completion == "complete", view.needs_attention
            trials = (await run.trials()).items
            assert {trial.candidate_slot_id for trial in trials} == {"original", "selected"}
            for trial in trials:
                graph = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
                execution = await graph.execution("target")
                request = str((await execution.model_interactions(include_content=True)).items[0].request)
                expected_prompt = original_prompt if trial.candidate_slot_id == "original" else candidate_prompt
                other_prompt = candidate_prompt if trial.candidate_slot_id == "original" else original_prompt
                assert request.count(expected_prompt) == 1
                assert other_prompt not in request
                assert "historical question" in request and "fixture answer" in request
                assert "captured question" in request
                if trial.candidate_slot_id == "selected":
                    retry = await execution.retry("replacement question")
                    assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
                    retry_request = str((await retry.model_interactions(include_content=True)).items[0].request)
                    assert retry_request.count(candidate_prompt) == 1
                    assert original_prompt not in retry_request
                    assert "historical question" in retry_request and "fixture answer" in retry_request
                    assert "replacement question" in retry_request
                    assert "captured question" not in retry_request
