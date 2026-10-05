#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured conversation input uses the selected candidate's Agent behavior."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, StartEvaluationRequest,
)
from linktools.ai.runtime import AgentTaskInput, CaptureInputRequest, ExecutionInputContext, Runtime, RuntimeStorage
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
    tmp_path: Path, source_kind: str, input_mode: str, target_kind: str,
) -> None:
    models = FixtureModels()
    group = CapabilityGroup("fork-candidate")
    original_prompt = "Original Agent behavior."
    candidate_prompt = "Selected candidate behavior."
    for name, prompt in (("default", original_prompt), ("candidate", candidate_prompt)):
        group.agent(name, model="default", system_prompt=prompt,
                    allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("fork-candidate", models=models, context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
        if source_kind == "fork":
            first = await runtime.agents.get().start("historical question", principal=PRINCIPAL)
            assert (await first.wait()).status is ExecutionStatus.SUCCEEDED
            source = await first.fork("captured question")
            assert (await source.wait()).status is ExecutionStatus.SUCCEEDED
        else:
            context = ExecutionInputContext.from_messages((
                ModelRequest(parts=[SystemPromptPart(original_prompt), UserPromptPart("historical question")]),
                ModelResponse(parts=[TextPart("fixture answer")]),
            ))
            source_task = runtime.tasks.from_agent("candidate.source", runtime.agents.get())
            graph = await runtime.tasks.bind(source_task).start(TaskGraph("source", (
                TaskNode("agent", task=source_task, input=AgentTaskInput("captured question", input_context=context)),
            )), principal=PRINCIPAL, idempotency_key="source")
            assert (await graph.wait()).status is TaskStatus.SUCCEEDED
            source = await graph.execution("agent")
        native_request = str((await source.model_interactions(include_content=True)).items[0].request)
        assert native_request.count(original_prompt) == 1
        assert candidate_prompt not in native_request
        capture = await runtime.executions.capture_input(source.execution_id, CaptureInputRequest(PRINCIPAL, "capture-fork"))
        original = await runtime.tasks.from_agent_capture("candidate.original", capture, principal=PRINCIPAL)
        selected = runtime.tasks.from_agent("candidate.selected", runtime.agents.get("candidate"))
        scorer = Task("candidate.score", exact, effect_policy="none")
        engine = runtime.tasks.bind(original, selected, scorer)
        case_capture = (await runtime._input_captures.task_input(capture, principal=PRINCIPAL)
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
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("fork-input", 1), (
            case,
        )), principal=PRINCIPAL, idempotency_key="publish-fork")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            candidates,
            (rule_scorer(scorer),), input_mode=input_mode, policy=EvaluationPolicy(model_fixtures=(models.contract,))),
            PRINCIPAL, "evaluate-fork"), engine=engine)
        view = await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)
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
                assert (await retry.wait()).status is ExecutionStatus.SUCCEEDED
                retry_request = str((await retry.model_interactions(include_content=True)).items[0].request)
                assert retry_request.count(candidate_prompt) == 1
                assert original_prompt not in retry_request
                assert "historical question" in retry_request and "fixture answer" in retry_request
                assert "replacement question" in retry_request
                assert "captured question" not in retry_request
