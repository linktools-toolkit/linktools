#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured input reprojection retains inline workspace attachment occurrences."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, UserContent

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import TaskStatus, WorkspaceFileInput
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, GraphTargetSpec, StartEvaluationRequest,
)
from linktools.ai.runtime import (
    AgentTaskInput, AgentTaskInputContext, CaptureGraphRequest, CaptureInputRequest,
    Runtime, RuntimeStorage,
)
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.workspace import Workspace
from .test_captured_graph_execution_inputs import _score_graph
from .test_evaluation_consumers import CONTEXT, PRINCIPAL, FixtureModels, rule_scorer


@pytest.mark.asyncio
@pytest.mark.parametrize(("capture_kind", "projected", "file_change"), [
    ("declaration_graph", False, "changed"),
    ("declaration_graph", True, "deleted"),
    ("materialized_graph", True, "changed"),
    ("materialized_graph", False, "deleted"),
    ("agent", False, "changed"),
    ("agent", True, "deleted"),
    ("task", True, "changed"),
    ("task", False, "deleted"),
])
async def test_capture_reprojects_inline_and_file_attachments_without_live_reads(
    tmp_path: Path, capture_kind: str, projected: bool, file_change: str,
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    inline = work / "inline.txt"
    inline.write_text("INLINE ORIGINAL", encoding="utf-8")
    suffix = work / "suffix.txt"
    suffix.write_text("SUFFIX ORIGINAL", encoding="utf-8")
    models = FixtureModels()
    group = CapabilityGroup("inline-capture")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def project(context: AgentTaskInputContext) -> tuple[UserContent | WorkspaceFileInput, ...]:
        assert not isinstance(context.prompt, str)
        return (f"Prepared in {context.graph_id}: ", *context.prompt)

    async with Runtime.open("inline-capture", models=models, context=CONTEXT,
            storage=RuntimeStorage.filesystem(tmp_path / "state"),
            capabilities=(group, CapabilityGroup("work", workspace=Workspace.load(work)))) as runtime:
        target = runtime.tasks.from_agent("inline.agent", runtime.agents.get(),
                                          build_input=project if projected else None)
        scorer = Task("inline.score", _score_graph, effect_policy="none")
        engine = runtime.tasks.bind(target, scorer)
        prompt = ("Before", BinaryContent(b"DIRECT", media_type="text/plain"),
                  WorkspaceFileInput("./inline.txt", identifier="first"), "Between",
                  WorkspaceFileInput("inline.txt", identifier="second"), "After")
        files = ("suffix.txt",) if file_change == "deleted" else ()
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=target, input=AgentTaskInput(prompt, files=files)),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
        expected = [b"DIRECT", b"INLINE ORIGINAL", b"INLINE ORIGINAL"]
        if files:
            expected.append(b"SUFFIX ORIGINAL")
        assert models.attachments == expected
        if capture_kind.endswith("graph"):
            capture = await runtime.tasks.capture_graph(source.graph_id,
                CaptureGraphRequest(PRINCIPAL, "capture", mode=capture_kind, context_policy="clean"))
            case = CaseSpec.graph(CaseRef("data", "case", 1), inputs={})
            candidate = CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "agent"}))
        else:
            execution = await source.execution("agent")
            capture = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(PRINCIPAL, "capture", context_policy="clean"))
            if capture_kind == "task":
                capture = await runtime._input_captures.task_input(capture, principal=PRINCIPAL)
            case = CaseSpec.from_capture(CaseRef("data", "case", 1), capture=capture)
            candidate = CandidateSpec("captured", task=target.ref)
        for file in (inline, suffix):
            file.unlink() if file_change == "deleted" else file.write_text("MUTATED", encoding="utf-8")
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (case,)),
            principal=PRINCIPAL, idempotency_key="dataset")
        for input_mode in ("fixed_input", "reproject_input"):
            run = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (candidate,), (rule_scorer(scorer),), input_mode=input_mode,
                policy=EvaluationPolicy(model_fixtures=(models.contract,))),
                PRINCIPAL, input_mode), engine=engine)
            assert (await run.wait(timeout_seconds=20)).completion == "complete"
            assert (await run.report()).scores[0].valid == 1
            assert models.attachments[-len(expected):] == expected
        assert models.attachments == expected * 3
        assert models.prompts[1] == models.prompts[0]
        assert (models.prompts[2] != models.prompts[0]) == projected


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_kind", ["declaration_graph", "materialized_graph", "agent", "task"])
async def test_irreversible_projection_keeps_fixed_capture_and_rejects_reprojection(
    tmp_path: Path, capture_kind: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    work = tmp_path / "work"
    work.mkdir()
    attachment = work / "unused.txt"
    attachment.write_text("NOT ACCEPTED", encoding="utf-8")
    models = FixtureModels()
    group = CapabilityGroup("irreversible-capture")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())

    async def discard(context: AgentTaskInputContext) -> str:
        return "Projected without the original attachment"

    async with Runtime.open("irreversible-capture", models=models, context=CONTEXT,
            storage=RuntimeStorage.filesystem(tmp_path / "state"),
            capabilities=(group, CapabilityGroup("work", workspace=Workspace.load(work)))) as runtime:
        target = runtime.tasks.from_agent("irreversible.agent", runtime.agents.get(), build_input=discard)
        scorer = Task("irreversible.score", _score_graph, effect_policy="none")
        engine = runtime.tasks.bind(target, scorer)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=target, input=AgentTaskInput((WorkspaceFileInput("unused.txt"),))),
        )), principal=PRINCIPAL, idempotency_key="source")
        assert (await source.wait(timeout_seconds=15)).status is TaskStatus.SUCCEEDED
        assert models.attachments == []
        if capture_kind.endswith("graph"):
            capture = await runtime.tasks.capture_graph(source.graph_id,
                CaptureGraphRequest(PRINCIPAL, "capture", mode=capture_kind, context_policy="clean"))
            case = CaseSpec.graph(CaseRef("data", "case", 1), inputs={})
            candidate = CandidateSpec("captured", graph_template=GraphTargetSpec(capture=capture, outputs={"answer": "agent"}))
        else:
            execution = await source.execution("agent")
            capture = await runtime.executions.capture_input(execution.execution_id,
                CaptureInputRequest(PRINCIPAL, "capture", context_policy="clean"))
            if capture_kind == "task":
                capture = await runtime._input_captures.task_input(capture, principal=PRINCIPAL)
            case = CaseSpec.from_capture(CaseRef("data", "case", 1), capture=capture)
            candidate = CandidateSpec("captured", task=target.ref)
        attachment.unlink()
        dataset = await runtime.evaluations.publish_dataset(DatasetSpec(DatasetRef("data", 1), (case,)),
            principal=PRINCIPAL, idempotency_key="dataset")
        fixed = await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
            (candidate,), (rule_scorer(scorer),), policy=EvaluationPolicy(model_fixtures=(models.contract,))),
            PRINCIPAL, "fixed"), engine=engine)
        assert (await fixed.wait(timeout_seconds=20)).completion == "complete"
        assert (await fixed.report()).scores[0].valid == 1
        with pytest.raises(AIError) as raised:
            await runtime.evaluations.start(StartEvaluationRequest(EvaluationSpec(dataset,
                (candidate,), (rule_scorer(scorer),), input_mode="reproject_input",
                policy=EvaluationPolicy(model_fixtures=(models.contract,))),
                PRINCIPAL, "reproject"), engine=engine)
        assert raised.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
        assert raised.value.safe_details["reason"] == "accepted_workspace_input_not_retained"
        assert models.prompts == ["Projected without the original attachment"] * 2
