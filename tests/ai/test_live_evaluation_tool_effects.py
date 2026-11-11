#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation effect limits follow live tools, graph children, and durable admission."""

from pathlib import Path
from typing import Literal

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, ExecutionCapability, MemoryCapability
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus, canonical_sha256
from linktools.ai.evaluation import (
    CandidateSpec, CaseRef, CaseSpec, DatasetRef, DatasetSpec, EvaluationPolicy,
    EvaluationSpec, StartEvaluationRequest,
)
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.errors import ErrorCode
from linktools.ai.execution import ExecutionInputContext, ForkExecutionRequest, RetryExecutionRequest
from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import MCPServerSpec
from linktools.ai.task import Task, TaskExpander, TaskGraph, TaskGraphSubmission, TaskNode, TaskNodeContext
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_current_toolsets import _fake_mcp
from .test_evaluation_consumers import (
    CONTEXT, EVALUATION_COMPLETION_TIMEOUT_SECONDS, PRINCIPAL, TENANT, exact, rule_scorer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,addition", (
    ("deny", "none"), ("read_only", "none"), ("read_only", "tool"),
    ("read_only", "mcp"), ("live", "tool"),
))
async def test_evaluation_enforces_effect_limit_against_next_step_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    policy: Literal["deny", "read_only", "live"], addition: str,
) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.agent("default", allow_skills=(), allow_subagents=())
    effects: list[str] = []
    mcp_events: list[str] = []
    mcp_instances = _fake_mcp(monkeypatch, mcp_events)

    def write(context: AgentContext[None]) -> str:
        effects.append("write")
        return "written"

    def read(context: AgentContext[None]) -> str:
        effects.append("read")
        if addition == "tool":
            features.tool(write, effect_policy="replay_safe")
        elif addition == "mcp":
            features.mcp(MCPServerSpec("late", transport="streamable-http", url="https://example.invalid/mcp"))
        return "read"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returned:
            return ModelResponse(parts=[ToolCallPart("read", {}, "read-call")])
        if addition == "tool" and not any(part.tool_name == "write" for part in returned):
            return ModelResponse(parts=[ToolCallPart("write", {}, "write-call")])
        return ModelResponse(parts=[TextPart("done")])

    models = _Models(_UsageFunctionModel(respond))
    features.model("default", models)
    features.tool(read, effect_policy="none")
    async with (
        Persistence.open("live-effects", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        target = Task.from_agent("effects.target", runtime.agents.get())
        scorer = Task("effects.score", exact, effect_policy="none")
        async with (
            TaskEngine.open(target, scorer, persistence=state, context=CONTEXT) as engine,
            Evaluations.open(engine) as evaluations,
        ):
            dataset = await evaluations.publish_dataset(DatasetSpec(DatasetRef("effects", 1), cases=(
                CaseSpec.agent(CaseRef("effects", "one", 1), prompt="go", expected={"text": "done"}),
            )), principal=PRINCIPAL, idempotency_key="dataset")
            request = StartEvaluationRequest(EvaluationSpec(
                dataset, (CandidateSpec("candidate", task=target.ref),), (rule_scorer(scorer),),
                policy=EvaluationPolicy(external_effects=policy, model_fixtures=(models.contract,)),
            ), PRINCIPAL, "evaluate")
            run = await evaluations.start(request)
            await run.wait(timeout_seconds=EVALUATION_COMPLETION_TIMEOUT_SECONDS)
            trial = (await run.trials()).items[0]
            blocked = policy == "deny" or (policy == "read_only" and addition != "none")
            assert trial.execution_status is (ExecutionStatus.FAILED if blocked else ExecutionStatus.SUCCEEDED)
            if blocked:
                assert trial.error_code == ErrorCode.EVALUATION_INCOMPATIBLE.value
            assert effects == ([] if policy == "deny" else ["read", "write"] if policy == "live" else ["read"])
            assert not mcp_instances and not mcp_events


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen", (False, True))
async def test_expanded_agent_inherits_durable_graph_tool_limit(tmp_path: Path, reopen: bool) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.agent("default", allow_skills=(), allow_subagents=())
    effects: list[str] = []

    def write(context: AgentContext[None]) -> str:
        effects.append("write")
        return "written"

    async def seed(context: TaskNodeContext[None]) -> JsonValue:
        features.tool(write, effect_policy="replay_safe")
        return None

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("write", {}, "write-call")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with (
        Persistence.open("expanded-effects", storage=storage, tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        child = Task.from_agent("effects.child", runtime.agents.get())
        initial = Task("effects.seed", seed, effect_policy="none")
        expander = TaskExpander("effects.expand", lambda context: (
            TaskNode("child", ("seed",), task=child, input=AgentTaskInput("go")),
        ))
        async with TaskEngine.open(initial, child, expander, persistence=state, context=CONTEXT) as engine:
            submission = await engine.prepare_submission(TaskGraph("expanded", (
                TaskNode("seed", task=initial, expander=expander.ref),
            )), principal=PRINCIPAL, idempotency_key="prepare", tool_effects="read_only")
            submission = decode_domain(encode_domain(submission), TaskGraphSubmission)
            assert submission.admission.tool_effects == "read_only"
            if not reopen:
                await engine.start_prepared(submission)
                run = await engine.get("expanded", principal=PRINCIPAL)
                assert (await run.wait(timeout_seconds=20)).result.status is TaskStatus.FAILED
                execution = await run.execution("child")
                outcome = (await execution.wait()).result
                assert outcome.status is ExecutionStatus.FAILED
                assert outcome.error_code == ErrorCode.EVALUATION_INCOMPATIBLE.value
    if reopen:
        async with (
            Persistence.open("expanded-effects", storage=storage, tenant_id=TENANT) as state,
            Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
        ):
            child = Task.from_agent("effects.child", runtime.agents.get())
            expander = TaskExpander("effects.expand", lambda context: (
                TaskNode("child", ("seed",), task=child, input=AgentTaskInput("go")),
            ))
            async with TaskEngine.open(initial, child, expander, persistence=state, context=CONTEXT) as engine:
                await engine.start_prepared(submission)
                run = await engine.get("expanded", principal=PRINCIPAL)
                assert (await run.wait(timeout_seconds=20)).result.status is TaskStatus.FAILED
                execution = await run.execution("child")
                record = await state.execution.executions.get(execution.execution_id, tenant_id=TENANT)
                assert record is not None and record.tool_effects == "read_only"
    assert effects == []


@pytest.mark.asyncio
async def test_subagent_inherits_parent_tool_limit_for_live_child_declarations(tmp_path: Path) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=("child",))
    features.agent("child", allow_skills=(), allow_subagents=())
    effects: list[str] = []

    def write(context: AgentContext[None]) -> str:
        effects.append("write")
        return "written"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        names = {tool.name for tool in info.function_tools}
        returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if "delegate_task" in names:
            if returned:
                return ModelResponse(parts=[TextPart("child finished")])
            features.tool(write, effect_policy="replay_safe")
            return ModelResponse(parts=[ToolCallPart("delegate_task", {
                "subagent_id": "child", "task": "go",
            }, "delegate-call")])
        if returned:
            return ModelResponse(parts=[TextPart("written")])
        return ModelResponse(parts=[ToolCallPart("write", {}, "write-call")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    async with (
        Persistence.open("child-effects", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        target = Task.from_agent("effects.parent", runtime.agents.get())
        async with TaskEngine.open(target, persistence=state, context=CONTEXT) as engine:
            submission = await engine.prepare_submission(TaskGraph("parent", (
                TaskNode("target", task=target, input=AgentTaskInput("delegate")),
            )), principal=PRINCIPAL, idempotency_key="prepare", tool_effects="read_only")
            await engine.start_prepared(submission)
            run = await engine.get("parent", principal=PRINCIPAL)
            await run.wait(timeout_seconds=20)
            parent = await run.execution("target")
            children = await runtime.executions.list_children(parent.execution_id, principal=PRINCIPAL)
            assert len(children) == 1
            child = await state.execution.executions.get(children[0].execution_id, tenant_id=TENANT)
            assert child is not None and child.tool_effects == "read_only"
            assert child.status is ExecutionStatus.FAILED
            assert child.error_code == ErrorCode.EVALUATION_INCOMPATIBLE.value
            assert effects == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ("read_memory", "write_memory"))
async def test_read_only_graph_validates_selected_memory_tools(tmp_path: Path, tool_name: str) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.capability(MemoryCapability(scope="evaluation"))
    features.agent("default", allow_tools=(tool_name,), allow_skills=(), allow_subagents=())
    returned_tools: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returned:
            returned_tools.extend(part.tool_name for part in returned)
            return ModelResponse(parts=[TextPart("done")])
        arguments = {"file": "MEMORY.md"}
        if tool_name == "write_memory":
            arguments["content"] = "must not be persisted"
        return ModelResponse(parts=[ToolCallPart(tool_name, arguments, "memory-call")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    async with (
        Persistence.open("memory-effects", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        target = Task.from_agent("effects.memory", runtime.agents.get())
        async with TaskEngine.open(target, persistence=state, context=CONTEXT) as engine:
            submission = await engine.prepare_submission(TaskGraph("memory", (
                TaskNode("target", task=target, input=AgentTaskInput("go")),
            )), principal=PRINCIPAL, idempotency_key="prepare", tool_effects="read_only")
            await engine.start_prepared(submission)
            run = await engine.get("memory", principal=PRINCIPAL)
            expected = TaskStatus.SUCCEEDED if tool_name == "read_memory" else TaskStatus.FAILED
            assert (await run.wait(timeout_seconds=20)).result.status is expected
            assert returned_tools == (["read_memory"] if tool_name == "read_memory" else [])
            assert await state.memory.records.capture_scope(
                tenant_id=TENANT, memory_scope="evaluation",
            ) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy,tool_name", (
    ("read_only", "read_memory"), ("read_only", "write_memory"),
    ("deny", "read_memory"), ("live", "write_memory"),
))
async def test_imported_memory_obeys_effect_limit_without_current_memory_declaration(
    tmp_path: Path, policy: Literal["deny", "read_only", "live"], tool_name: str,
) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.agent("default", allow_tools=(tool_name,), allow_skills=(), allow_subagents=())
    importing = False
    returned_tools: list[ToolReturnPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not importing:
            return ModelResponse(parts=[TextPart("captured")])
        assert {tool.name for tool in info.function_tools} == {tool_name}
        returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returned:
            returned_tools.extend(returned)
            return ModelResponse(parts=[TextPart("done")])
        arguments = ({"file": "NEW.md", "content": "new memory"}
                     if tool_name == "write_memory" else {"file": "MEMORY.md"})
        return ModelResponse(parts=[ToolCallPart(tool_name, arguments, "memory-call")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    context = ExecutionInputContext.from_messages((), memory={
        "memory/MEMORY.md": {"content": "captured memory", "version": "m1:" + "1" * 64},
    })
    async with (
        Persistence.open("imported-effects", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        source = await runtime.agents.get().start("go", principal=PRINCIPAL, input_context=context)
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        target = Task.from_agent("effects.imported", runtime.agents.get())
        async with TaskEngine.open(target, persistence=state, context=CONTEXT) as engine:
            capture = await engine.inputs.capture_input(source.execution_id, CaptureInputRequest(PRINCIPAL, "capture"))
            captured = await engine.inputs.read_agent(capture, principal=PRINCIPAL)
            assert captured.input_context is not None and captured.input_context.memory == context.memory
            importing = True
            run = await engine.start(TaskGraph("imported", (
                TaskNode("target", task=target, input=AgentTaskInput(captured.prompt, input_context=captured.input_context)),
            )), principal=PRINCIPAL, idempotency_key="import", tool_effects=policy)
            blocked = policy == "deny" or (policy == "read_only" and tool_name == "write_memory")
            assert (await run.wait(timeout_seconds=20)).result.status is (TaskStatus.FAILED if blocked else TaskStatus.SUCCEEDED)
            execution = await run.execution("target")
            outcome = (await execution.wait()).result
            if blocked:
                assert outcome.error_code == ErrorCode.EVALUATION_INCOMPATIBLE.value
            assert [part.tool_name for part in returned_tools] == ([] if blocked else [tool_name])
            if tool_name == "read_memory" and not blocked:
                assert "captured memory" in str(returned_tools[0].content)
            record = await state.execution.executions.get(execution.execution_id, tenant_id=TENANT)
            assert record is not None and record.context_imported and record.memory_scope is not None
            assert record.tool_effects == policy
            records = await state.memory.records.capture_scope(
                tenant_id=TENANT, memory_scope=record.memory_scope,
            )
            assert sorted(item.metadata["path"] for item in records) == (
                ["memory/MEMORY.md", "memory/NEW.md"] if policy == "live" else ["memory/MEMORY.md"]
            )


@pytest.mark.asyncio
async def test_retry_and_fork_keep_tool_limit_while_new_runs_default_to_live(tmp_path: Path) -> None:
    features = RuntimeFeatures(workspace_factory=None)
    features.capability(ExecutionCapability(retain_history=True))
    features.agent("default", allow_skills=(), allow_subagents=())
    effects: list[str] = []

    def write(context: AgentContext[None]) -> str:
        effects.append("write")
        return "written"

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        if "write" in {tool.name for tool in info.function_tools} and not returned:
            return ModelResponse(parts=[ToolCallPart("write", {}, "write-call")])
        return ModelResponse(parts=[TextPart("done")])

    features.model("default", _Models(_UsageFunctionModel(respond)))
    async with (
        Persistence.open("retry-effects", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id=TENANT) as state,
        Runtime.attach(persistence=state, context=CONTEXT, features=features) as runtime,
    ):
        target = Task.from_agent("effects.source", runtime.agents.get())
        async with TaskEngine.open(target, persistence=state, context=CONTEXT) as engine:
            submission = await engine.prepare_submission(TaskGraph("source", (
                TaskNode("target", task=target, input=AgentTaskInput("go")),
            )), principal=PRINCIPAL, idempotency_key="prepare", tool_effects="read_only")
            await engine.start_prepared(submission)
            run = await engine.get("source", principal=PRINCIPAL)
            assert (await run.wait(timeout_seconds=20)).result.status is TaskStatus.SUCCEEDED
            source = await run.execution("target")
            features.tool(write, effect_policy="replay_safe")
            retry = await runtime.executions.retry(source.execution_id, RetryExecutionRequest("retry", PRINCIPAL, "retry"))
            fork = await runtime.executions.fork(source.execution_id, ForkExecutionRequest("fork", PRINCIPAL, "fork"))
            for handle in (retry, fork):
                outcome = await runtime.executions.wait(handle.execution_id, principal=PRINCIPAL, timeout_seconds=20)
                assert outcome.result.status is ExecutionStatus.FAILED
                assert outcome.result.error_code == ErrorCode.EVALUATION_INCOMPATIBLE.value
                record = await state.execution.executions.get(handle.execution_id, tenant_id=TENANT)
                assert record is not None and record.tool_effects == "read_only"
            assert effects == []
            ordinary = await runtime.agents.get().run("ordinary", principal=PRINCIPAL)
            assert ordinary.result.status is ExecutionStatus.SUCCEEDED
            assert effects == ["write"]
