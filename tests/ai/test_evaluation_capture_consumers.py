#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Historical child and graph captures run through the public evaluation API."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.agent import AgentInputCaptureRef
from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, ScoreBundle, ScoringInput, StartEvaluationRequest,
    TaskCaseInput,
)
from linktools.ai.runtime import CaptureGraphRequest, CaptureInputRequest, Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext, TaskNodeResultRef
from linktools.ai.workspace import Workspace

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, PRINCIPAL, FixtureModels, exact, rule_scorer


class DelegatingModels(FixtureModels):
    def __init__(self) -> None:
        super().__init__()
        self.delegations = 0
        self.child_prompts: list[str] = []

    def materialize(self) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            prompt = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)][-1]
            if prompt == "delegate this work":
                if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
                    yield "parent finished"
                else:
                    assert "delegate_task" in {tool.name for tool in info.function_tools}
                    self.delegations += 1
                    yield {0: DeltaToolCall(name="delegate_task", json_args=json.dumps({
                        "subagent_id": "child", "task": "the accepted child question",
                    }))}
            else:
                assert prompt == "the accepted child question"
                assert "delegate_task" not in {tool.name for tool in info.function_tools}
                self.child_prompts.append(prompt)
                yield "the child answer"
        return FunctionModel(stream_function=respond)


def child_capabilities(revision: int, instruction: str) -> CapabilityGroup[None]:
    group = CapabilityGroup[None]("captured-child-consumer")
    group.agent("default", model="default", system_prompt="Parent behavior", allow_tools=(),
                allow_skills=(), allow_subagents=("child",))
    group.agent("child", revision=revision, model="default", system_prompt=instruction, allow_tools=(),
                allow_skills=(), allow_subagents=("grandchild",))
    group.agent("grandchild", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return group


@pytest.mark.asyncio
async def test_tool_created_child_capture_runs_independently_in_evaluation(tmp_path: Path) -> None:
    models = DelegatingModels()
    work = tmp_path / "workspace"
    work.mkdir()
    instructions = work / "AGENTS.md"
    instructions.write_text("Historical child repository context", encoding="utf-8")
    state = tmp_path / "state"
    async with Runtime.open("child-capture-consumer", models=models, storage=RuntimeStorage.filesystem(state),
            context=CONTEXT, capabilities=(child_capabilities(1, "Historical child behavior"),
                CapabilityGroup("workspace", workspace=Workspace.load(work)))) as runtime:
        parent = await runtime.agents.get().start("delegate this work", principal=PRINCIPAL,
                                                idempotency_key="create-real-child")
        assert (await parent.wait(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        children = await runtime.executions.list_children(parent.execution_id, principal=PRINCIPAL)
        assert len(children) == models.delegations == 1
        source_view = children[0]
        assert source_view.agent_id == "child"
        assert source_view.lineage_kind is ExecutionLineageKind.SUBAGENT
        assert source_view.parent_execution_id == parent.execution_id
        assert source_view.parent_invocation_id is not None
        source_id = source_view.execution_id
        source_result = await runtime.executions.result(source_id, principal=PRINCIPAL)
        source_info = await runtime.history.inspect_execution(source_id, principal=PRINCIPAL)
        source_interactions = await runtime.executions.model_interactions(source_id, principal=PRINCIPAL, include_content=True)
        assert "Historical child behavior" in str(source_interactions.items[0].request)
        assert "Historical child repository context" in str(source_interactions.items[0].request)
        capture = await runtime.executions.capture_input(source_id, CaptureInputRequest(PRINCIPAL, "capture-tool-child"))
        assert isinstance(capture, AgentInputCaptureRef)

    instructions.write_text("Changed live repository context", encoding="utf-8")
    scorer = Task("consumer.captured-child-score", exact, effect_policy="none")
    async with Runtime.open("child-capture-consumer", models=models, storage=RuntimeStorage.filesystem(state),
            context=CONTEXT, capabilities=(child_capabilities(2, "Changed current child behavior"),
                CapabilityGroup("workspace", workspace=Workspace.load(work)))) as runtime:
        historical = await runtime.tasks.from_agent_capture("consumer.tool-child", capture, principal=PRINCIPAL)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("tool-child", 1), cases=(
            CaseSpec.from_capture(CaseRef("tool-child", "original", 1), capture=capture, expected=source_result.output),
        )), principal=PRINCIPAL, idempotency_key="publish-tool-child")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("historical-child", task=historical.ref),), (rule_scorer(scorer),),
            policy=EvaluationPolicy(model_fixtures=(models.contract,))), PRINCIPAL, "evaluate-tool-child"),
            engine=runtime.tasks.bind(historical, scorer))
        view = await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        assert view.completion == "complete", view.needs_attention
        report = await run.report()
        assert report.scores[0].valid == 1 and report.scores[0].mean == 1.0
        trial = (await run.trials()).items[0]
        replay_id = trial.subject.execution_id
        assert replay_id != source_id
        replay = await runtime.executions.inspect(replay_id, principal=PRINCIPAL)
        assert replay.agent_id == "child" and replay.status is ExecutionStatus.SUCCEEDED
        assert replay.lineage_kind is ExecutionLineageKind.RUN
        assert replay.parent_execution_id is replay.parent_invocation_id is replay.session_id is None
        assert replay.root_execution_id == replay_id
        replay_info = await runtime.history.inspect_execution(replay_id, principal=PRINCIPAL)
        assert replay_info.binding_digest == source_info.binding_digest
        replay_interactions = await runtime.executions.model_interactions(replay_id, principal=PRINCIPAL, include_content=True)
        request = str(replay_interactions.items[0].request)
        assert "the accepted child question" in request
        assert "Historical child behavior" in request and "Changed current child behavior" not in request
        assert "Historical child repository context" in request and "Changed live repository context" not in request
        evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
        assert evidence.target.output.value == source_result.output
        assert models.child_prompts == ["the accepted child question"] * 2
        assert models.delegations == 1
        assert await runtime.executions.inspect(source_id, principal=PRINCIPAL) == source_view
        assert await runtime.executions.result(source_id, principal=PRINCIPAL) == source_result
        assert await runtime.history.inspect_execution(source_id, principal=PRINCIPAL) == source_info
        assert await runtime.executions.model_interactions(source_id, principal=PRINCIPAL, include_content=True) == source_interactions


@pytest.mark.asyncio
async def test_graph_capture_ref_evaluates_new_inputs_after_reopen(tmp_path: Path) -> None:
    seen: list[str] = []

    async def prepare(context: TaskNodeContext[None]) -> JsonValue:
        question = context.input.get("question", context.input["fallback_question"])
        seen.append(question)
        return question.strip().upper()

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        return {"answer": await context.read_dependency("prepared")}

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        assert sample.target.node_statuses == {"prepare": "succeeded", "answer": "succeeded"}
        return ScoreBundle(dimensions={"exact_match": float(sample.target_output["answer"]["value"] == sample.expected)}).to_mapping()

    producer = Task("consumer.capture-prepare", prepare, effect_policy="none")
    consumer = Task("consumer.capture-answer", answer, effect_policy="none")
    scorer = Task("consumer.capture-graph-score", score, effect_policy="none")
    async with Runtime.open("graph-capture-ref", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        original = await runtime.tasks.bind(producer, consumer).start(TaskGraph("source-captured-graph", nodes=(
            TaskNode("prepare", task=producer, input={"fallback_question": "old source input"}),
            TaskNode("answer", ("prepare",), task=consumer, input_refs={"prepared": TaskNodeResultRef("prepare")}),
        )), principal=PRINCIPAL, idempotency_key="source-captured-graph")
        assert (await original.wait(timeout_seconds=10)).status is TaskStatus.SUCCEEDED
        original_view = await original.inspect()
        original_answer = await original.result("answer")
        original_ids = tuple([(await original.execution(node)).execution_id for node in ("prepare", "answer")])
        capture = await runtime.tasks.capture_graph(original.graph_id, CaptureGraphRequest(PRINCIPAL, "capture-whole-graph"))

    async with Runtime.open("graph-capture-ref", models=FixtureModels(), storage=RuntimeStorage.filesystem(tmp_path),
                            context=CONTEXT) as runtime:
        engine = runtime.tasks.bind(producer, consumer, scorer)
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("captured-graph-ref", 1), cases=(
            CaseSpec.graph(CaseRef("captured-graph-ref", "new", 1),
                inputs={"prepare": TaskCaseInput(input={"question": "new case input"})}, expected={"answer": "NEW CASE INPUT"}),
        )), principal=PRINCIPAL, idempotency_key="publish-captured-graph-ref")
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("captured-workflow", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "answer"})),),
            (rule_scorer(scorer),)), PRINCIPAL, "evaluate-captured-graph-ref"), engine=engine)
        view = await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)
        assert view.completion == "complete", view.needs_attention
        report = await run.report()
        assert report.scores[0].valid == 1 and report.scores[0].mean == 1.0
        trial = (await run.trials()).items[0]
        assert trial.graph_ref.graph_id != original.graph_id
        evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
        assert evidence.target.node_statuses == {"prepare": "succeeded", "answer": "succeeded"}
        assert evidence.target.outputs["answer"].value.value == {"answer": "NEW CASE INPUT"}
        rerun = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
        assert all([(await rerun.execution(node)).execution_id not in original_ids for node in ("prepare", "answer")])
        source = await engine.get(original.graph_id, principal=PRINCIPAL)
        assert await source.inspect() == original_view
        assert await source.result("answer") == original_answer == {"answer": "OLD SOURCE INPUT"}
        assert seen == ["old source input", "new case input"]
