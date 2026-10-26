#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured graph replay preserves the native node's structured output contract."""

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations
from ._runtime_test_helpers import model_features

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, ScoreBundle, StartEvaluationRequest,
)
from linktools.ai.inputs import AgentTaskInput, CaptureGraphRequest
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, PRINCIPAL, FixtureModels, rule_scorer


class Answer(BaseModel):
    answer: str


class StructuredModels(FixtureModels):
    def materialize(self) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            if info.output_tools:
                output = info.output_tools[0]
                self.schemas.append(output.parameters_json_schema)
                yield {0: DeltaToolCall(name=output.name, json_args=json.dumps({"answer": "yes"}))}
            else:
                yield "unstructured answer"
        return FunctionModel(stream_function=respond)


@pytest.mark.asyncio
async def test_captured_graph_keeps_node_output_contract_after_reopen(tmp_path: Path) -> None:
    async def score(context: TaskNodeContext[None]) -> JsonValue:
        return ScoreBundle(dimensions={"exact_match": 1}).to_mapping()

    models = StructuredModels()
    group = RuntimeFeatures("structured-graph")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    scorer = Task("graph.score", score, effect_policy="none")
    async with (
        Persistence.open('structured-graph', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_53,
        Runtime.attach(persistence=persistence_53, context=CONTEXT, features=model_features(models, group)) as runtime,
    ):
        target = Task.from_agent("graph.target", runtime.agents.get())
        async with TaskEngine.open(target, persistence=persistence_53, context=CONTEXT) as engine:
            source = await engine.start(TaskGraph("source", (
                TaskNode("answer", task=target, input=AgentTaskInput("question"), output_type=Answer),
            )), principal=PRINCIPAL, idempotency_key="source")
            assert (await source.wait(timeout_seconds=10)).result.wait_status is TaskStatus.SUCCEEDED
            source_output = await source.result("answer")
            source_contract = (await source.state(include_content=True)).nodes[0].output_contract
            capture = await engine.inputs.capture_graph(source.graph_id, CaptureGraphRequest(PRINCIPAL, "capture"))
    async with (
        Persistence.open('structured-graph', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_63,
        Runtime.attach(persistence=persistence_63, context=CONTEXT, features=model_features(models, group)) as runtime,
    ):
        target = Task.from_agent("graph.target", runtime.agents.get())
        async with (
            TaskEngine.open(target, scorer, persistence=persistence_63, context=CONTEXT) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (
                CaseSpec.graph(CaseRef("data", "one", 1), inputs={}),
            )), principal=PRINCIPAL, idempotency_key="dataset")
            run = await evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "answer"})),),
                (rule_scorer(scorer),), policy=EvaluationPolicy(model_fixtures=(models.contract,))),
                PRINCIPAL, "evaluate"))
            assert (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
            trial = (await run.trials()).items[0]
            evidence = await evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
            assert evidence.target.outputs["answer"].value.value == source_output == {"answer": "yes"}
            replay = await engine.get(trial.graph_ref.graph_id, principal=PRINCIPAL)
            assert (await replay.state(include_content=True)).nodes[0].output_contract == source_contract


@pytest.mark.asyncio
async def test_resolved_output_contract_must_match_named_task_schema(tmp_path: Path) -> None:
    class DifferentAnswer(BaseModel):
        different: int

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        return {"answer": "yes"}

    declared = Task("schema.answer", answer, output_type=Answer, effect_policy="none")
    different = Task("schema.different", answer, output_type=DifferentAnswer, effect_policy="none")
    async with (
        Persistence.open('schema-contract', storage=PersistenceStorage.filesystem(tmp_path), tenant_id='evaluation-consumers') as persistence_92,
        TaskEngine.open(declared, persistence=persistence_92, context=CONTEXT) as engine,
    ):
        for task in (declared, different):
            contract = {"mode": "structured", "schema": task.contract["output_contract"]["schema"]}
            graph = TaskGraph(task.id, (TaskNode.from_resolved("answer", task=declared.ref, output_contract=contract),))
            if task is different:
                with pytest.raises(AIError) as rejected:
                    await engine.describe_submission(graph, principal=PRINCIPAL, idempotency_key=task.id)
                assert rejected.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
            else:
                submission = await engine.describe_submission(graph, principal=PRINCIPAL, idempotency_key=task.id)
                assert submission.graph.nodes[0].output_contract == contract
