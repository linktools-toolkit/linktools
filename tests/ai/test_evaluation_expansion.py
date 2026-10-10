#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic evaluation targets admit and recover their complete bound scope."""

from pathlib import Path

import pytest

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy, EvaluationSpec,
    GraphTargetSpec, ScoreBundle, ScoringInput, StartEvaluationRequest,
)
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.task import Task, TaskExpander, TaskGraphTemplate, TaskNode, TaskNodeContext

from .test_evaluation_consumers import EVALUATION_COMPLETION_TIMEOUT_SECONDS, CONTEXT, PRINCIPAL, FixtureModels, exact, rule_scorer


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_kind,external_effects", (
    ("non_replay_safe", "deny"), ("model", "deny"),
    ("replay_safe", "deny"), ("replay_safe", "read_only"),
    ("non_replay_safe", "live"), ("model", "live"),
))
async def test_dynamic_target_rejects_unsafe_bound_definition_before_callbacks(
    tmp_path: Path, unsafe_kind: str, external_effects: str,
) -> None:
    calls: list[str] = []

    async def seed(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("seed")
        return None

    async def effect(context: TaskNodeContext[None]) -> JsonValue:
        calls.append("effect")
        return None

    models = FixtureModels()
    group = CapabilityGroup[None]("expanded-model")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    initial = Task("expansion.seed", seed, effect_policy="none")
    scorer = Task("expansion.score", exact, effect_policy="none")
    async with Runtime.open("unsafe-expansion", models=models, context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
        unsafe = (Task("expansion.effect", effect, effect_policy=unsafe_kind) if unsafe_kind != "model"
                  else runtime.tasks.from_agent("expansion.model", runtime.agents.get()))
        expander = TaskExpander("expansion.expand", lambda context: (TaskNode("child", task=unsafe),))
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("expansion", 1), cases=(
            CaseSpec.graph(CaseRef("expansion", "one", 1), inputs={}),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        template = TaskGraphTemplate((TaskNode("seed", task=initial, expander=expander.ref),))
        with pytest.raises(AIError) as rejected:
            await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (CandidateSpec("expanded", graph_template=GraphTargetSpec(template=template, selector="terminal_sinks")),),
                (rule_scorer(scorer),), policy=EvaluationPolicy(external_effects=external_effects)),
                PRINCIPAL, "run"), engine=runtime.tasks.bind(initial, unsafe, scorer, expander))
        assert rejected.value.code is ErrorCode.EVALUATION_INCOMPATIBLE
        assert calls == [] and models.prompts == []


@pytest.mark.asyncio
async def test_safe_nested_expansion_recovers_only_its_pinned_definition_scope(tmp_path: Path) -> None:
    calls: list[str] = []

    async def answer(context: TaskNodeContext[None]) -> JsonValue:
        calls.append(context.input["answer"])
        return context.input["answer"]

    async def score(context: TaskNodeContext[None]) -> JsonValue:
        sample = ScoringInput.from_mapping(context.input)
        return ScoreBundle(dimensions={"exact_match": float(sample.target_output["leaf"]["value"] == sample.expected)}).to_mapping()

    async def cancel(context: TaskNodeContext[None]) -> None:
        calls.append("cancel")

    seed = Task("expansion.seed", answer, effect_policy="none")
    child = Task("expansion.child", answer, effect_policy="none")
    unused = Task("expansion.unused", answer, effect_policy="none")
    scorer = Task("expansion.score", score, effect_policy="none")
    nested = TaskExpander("expansion.nested", lambda context: (
        TaskNode("leaf", ("child",), task=child, input={"answer": "yes"}),))
    outer = TaskExpander("expansion.outer", lambda context: (
        TaskNode("child", ("seed",), task=child, input={"answer": "middle"}, expander=nested.ref),))
    definitions = (seed, child, unused, scorer, outer, nested)
    async with Runtime.open("safe-expansion", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("expansion", 1), cases=(
            CaseSpec.graph(CaseRef("expansion", "one", 1), inputs={}, expected="yes"),
        )), principal=PRINCIPAL, idempotency_key="dataset")
        template = TaskGraphTemplate((TaskNode("seed", task=seed, input={"answer": "seed"}, expander=outer.ref),))
        run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (CandidateSpec("expanded", graph_template=GraphTargetSpec(template=template, selector="terminal_sinks")),),
            (rule_scorer(scorer),)), PRINCIPAL, "run"), engine=runtime.tasks.bind(*definitions))
        view = (await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result
        assert view.completion == "complete", view.needs_attention
        assert (await run.create_report()).scores[0].mean == 1.0
        trial = (await run.trials()).items[0]
        assert trial.execution_status is TaskStatus.SUCCEEDED
        evidence = await runtime.evaluations.read_evidence(trial.evidence_ref, principal=PRINCIPAL)
        assert evidence.target.node_statuses == {"seed": "succeeded", "child": "succeeded", "leaf": "succeeded"}
        assert evidence.target.outputs["leaf"].value.value == "yes"
        experiment_id = run.experiment_id
    async with Runtime.open("safe-expansion", models=FixtureModels(), context=CONTEXT,
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        changed_task = Task(unused.id, answer, effect_policy="none", cancel=cancel)
        changed_expander = TaskExpander(nested.id, nested.expand, revision=2)
        added_task = Task("expansion.added", answer, effect_policy="non_replay_safe")
        replacements = (
            (seed, child, changed_task, scorer, outer, nested),
            (*definitions, added_task),
            (seed, child, unused, scorer, outer, changed_expander),
        )
        for index, replacement in enumerate(replacements):
            with pytest.raises(AIError) as rejected:
                await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(*replacement),
                    principal=PRINCIPAL, idempotency_key=f"changed-{index}")
            assert rejected.value.code is ErrorCode.BINDING_CONFLICT
        recovered = await runtime.evaluations.reconcile(experiment_id, engine=runtime.tasks.bind(*definitions),
            principal=PRINCIPAL, idempotency_key="unchanged")
        assert (await recovered.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)).result.completion == "complete"
        assert calls == ["seed", "middle", "yes"]
