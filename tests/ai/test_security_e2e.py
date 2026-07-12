#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end smoke tests for the managed security governance path. Verifies
the full chain: Runtime.build(security=...) -> AgentRunner.execute ->
ManagedToolAdapter -> pipeline before/after_tool -> handler -> result."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.errors import RunPaused
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime
from linktools.ai.security.baseline import SecurityBaseline
from linktools.ai.security.pipeline import (
    PipelineAction,
    PipelineDecision,
    ToolInvocationEvent,
    ToolResultEvent,
)
from linktools.ai.storage.facade import FileStorage


class _DenyAllPipeline:
    """Pipeline that denies every tool invocation, recording whether before_tool
    actually fired so a test can assert the pipeline is wired into the real
    execution path (not just stored on a field)."""

    def __init__(self) -> None:
        self.saw_before = False

    async def before_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def after_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def before_tool(self, e: ToolInvocationEvent):
        self.saw_before = True
        return PipelineDecision(
            action=PipelineAction.DENY, reason="blocked by e2e pipeline"
        )

    async def after_tool(self, e: ToolResultEvent):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def on_security_event(self, e):
        return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


def _deny_terminal_pipeline() -> _DenyAllPipeline:
    return _DenyAllPipeline()


def _model_fn_no_tools(messages, info: AgentInfo) -> ModelResponse:
    """Model that just returns text without calling tools."""
    return ModelResponse(parts=[TextPart(content="hello")])


def _registry():
    from linktools.ai.model.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn_no_tools))
    return registry


def _spec():
    return AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(),
    )


@pytest.mark.asyncio
async def test_runtime_with_default_baseline_runs(tmp_path):
    """Default SecurityBaseline does not break normal execution."""
    from linktools.ai.model.router import ModelRouter

    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
    )
    result = await rt.run(_spec(), "hello")
    assert "hello" in str(result.output)


@pytest.mark.asyncio
async def test_runtime_baseline_disabled_runs(tmp_path):
    """SecurityBaseline(enabled=False) falls back to legacy path."""
    from linktools.ai.model.router import ModelRouter

    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
        security=SecurityBaseline(enabled=False),
    )
    result = await rt.run(_spec(), "hello")
    assert "hello" in str(result.output)


@pytest.mark.asyncio
async def test_default_baseline_denies_dangerous_command_without_pause_on_approval(
    tmp_path,
):
    """Regression for the contract fix: Runtime.build's default SecurityBaseline
    (enabled=True) must reach the compiler's ToolExecutor even when
    pause_on_approval=False (the default) -- previously AgentCompiler silently
    built its OWN default denylist in that case, independent of ``security=``.
    Verified end-to-end: a real model bash("rm -rf /") call is denied (the
    command never executes), not just that a field is wired."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="bash", args={"command": "rm -rf /"})]
            )
        return ModelResponse(parts=[TextPart(content=_bash_outcome(messages))])

    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(model_fn))
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=LocalExecutionBackend(runtime_dir=tmp_path),
        options=CapabilityRuntimeOptions(
            tool_exposure=CapabilityToolExposurePolicy(expose_execution_tools=True)
        ),
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="terminal"),),
    )
    result = await rt.run(spec, "wipe it")
    # The dangerous command was denied by the default baseline denylist before
    # it could execute -- the model saw the denial, not a successful run.
    assert "BLOCKED" in str(result.output)


@pytest.mark.asyncio
async def test_disabled_baseline_actually_disables_denylist_without_pause_on_approval(
    tmp_path,
):
    """The exact bug contract closes: SecurityBaseline(enabled=False) passed to
    Runtime.build (pause_on_approval at its False default) must genuinely
    produce zero command rules -- not have AgentCompiler reinstate its own
    default denylist underneath. Verified end-to-end: a real model bash("echo
    ok") call actually executes (the baseline does not deny it)."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="bash", args={"command": "echo ok"})]
            )
        return ModelResponse(parts=[TextPart(content=_bash_outcome(messages))])

    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(model_fn))
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=LocalExecutionBackend(runtime_dir=tmp_path),
        security=SecurityBaseline(enabled=False),
        options=CapabilityRuntimeOptions(
            tool_exposure=CapabilityToolExposurePolicy(expose_execution_tools=True)
        ),
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="terminal"),),
    )
    result = await rt.run(spec, "say ok")
    # No denylist wired -> the safe command actually executed.
    assert "RAN" in str(result.output)


def _bash_outcome(messages) -> str:
    """Inspect the bash tool-return in the message history: 'BLOCKED' if the
    call was denied (error surfaced), 'RAN' if it executed successfully."""
    for m in messages:
        for p in getattr(m, "parts", []):
            if getattr(p, "part_kind", None) == "tool-return":
                content = str(getattr(p, "content", ""))
                low = content.lower()
                if "denied" in low or "error" in low or "skip" in low:
                    return "BLOCKED"
                return "RAN"
    return "NO_TOOL_RETURN"


@pytest.mark.asyncio
async def test_pipeline_attached_to_runtime(tmp_path):
    """A SecurityPipeline on the SecurityBaseline is wired into the runner and
    actually fires on a real tool call -- verified by behavior (the pipeline's
    before_tool hook runs and denies the call), not by inspecting a private
    runner field. contract forbids the latter."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter

    pipeline = _deny_terminal_pipeline()

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) <= 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": "x"})]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(model_fn))
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=LocalExecutionBackend(runtime_dir=tmp_path),
        security=SecurityBaseline(pipeline=pipeline),
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="file"),),
    )
    result = await rt.run(spec, "read it")
    # The pipeline's before_tool hook fired on the real tool call -- the
    # pipeline is genuinely wired into the execution path, not just stored.
    assert pipeline.saw_before is True
    # The call was denied before the handler ran; the model got the denial
    # surfaced back and continued to its text answer.
    assert "done" in str(result.output)


class _RequireApprovalPipeline:
    """Pipeline that requires approval for every tool call."""

    async def before_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def after_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def before_tool(self, e):
        return PipelineDecision(
            action=PipelineAction.REQUIRE_APPROVAL, reason="need ok"
        )

    async def after_tool(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def on_security_event(self, e):
        return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


def _tool_calling_model_fn(tool_name, args, final_text="COMPLETED"):
    """Model that calls a tool once, then returns final_text once it sees a
    tool-return."""

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        n = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if getattr(p, "part_kind", None) == "tool-return"
        )
        if n == 0:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content=final_text)])

    return model_fn


@pytest.mark.asyncio
async def test_managed_tool_approval_pauses_and_resumes_end_to_end(tmp_path):
    """The contract managed-approval closed loop: a managed tool whose approval
    originates from the pipeline (not the PolicyEngine) must (1) actually
    PAUSE the run -- not be swallowed into a skip-result by pydantic-ai's
    on_tool_execute_error -- and (2) complete on resume after approval without
    re-pausing. Regression: previously the pause was swallowed and the run
    completed having surfaced an error to the model."""
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter

    registry = ModelRegistry()
    registry.register(
        "m", model=FunctionModel(_tool_calling_model_fn("read_file", {"path": "x"}))
    )
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=LocalExecutionBackend(runtime_dir=tmp_path),
        security=SecurityBaseline(pipeline=_RequireApprovalPipeline()),
        options=CapabilityRuntimeOptions(tool_exposure=CapabilityToolExposurePolicy()),
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="file"),),
    )

    # Drive 1: the managed tool's pipeline-approval must PAUSE the run.
    with pytest.raises(RunPaused) as ei:
        await rt.run(spec, "read it")
    run_id = ei.value.run_id
    approval_id = ei.value.approval_id

    # Approve the paused request.
    req = await rt.storage.approvals.get(approval_id)
    await rt.storage.approvals.approve(
        approval_id, expected_version=req.version, resolved_by="tester"
    )

    # Resume: the re-driven call must NOT re-pause (already-approved gate) --
    # the run reaches a terminal state (SUCCEEDED), not WAITING_APPROVAL again.
    resumed_run = False
    async for _ev in rt.resume(run_id, spec):
        resumed_run = True
    record = await rt.storage.runs.get(run_id)
    from linktools.ai.run.models import RunStatus

    assert record is not None
    assert record.status is RunStatus.SUCCEEDED, (
        f"resume should succeed, not re-pause; got {record.status}"
    )
    assert resumed_run


class _ModifyPathPipeline:
    """Pipeline that rewrites the read_file `path` argument to a known file."""

    def __init__(self, target_path):
        self.target_path = target_path

    async def before_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def after_model(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def before_tool(self, e):
        return PipelineDecision(
            action=PipelineAction.MODIFY,
            modified_payload={**dict(e.arguments), "path": self.target_path},
        )

    async def after_tool(self, e):
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def on_security_event(self, e):
        return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


@pytest.mark.asyncio
async def test_pipeline_modify_arguments_e2e(tmp_path):
    """contract #9 e2e: a pipeline MODIFY actually changes the arguments the tool
    handler receives. The model calls read_file(path='wrong'); the pipeline
    rewrites path to a real file whose content the model then echoes back."""
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter

    (tmp_path / "real.txt").write_text("MODIFIED-CONTENT", encoding="utf-8")

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        n = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if getattr(p, "part_kind", None) == "tool-return"
        )
        if n == 0:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={"path": "wrong"})]
            )
        # Echo back whatever the tool returned.
        text = ""
        for m in messages:
            for p in getattr(m, "parts", []):
                if getattr(p, "part_kind", None) == "tool-return":
                    text = str(getattr(p, "content", ""))
        return ModelResponse(parts=[TextPart(content=text)])

    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(model_fn))
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=LocalExecutionBackend(runtime_dir=tmp_path),
        security=SecurityBaseline(pipeline=_ModifyPathPipeline("real.txt")),
        options=CapabilityRuntimeOptions(tool_exposure=CapabilityToolExposurePolicy()),
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="file-read"),),
    )
    result = await rt.run(spec, "read it")
    # The pipeline rewrote path -> real.txt, so the model saw its content.
    assert "MODIFIED-CONTENT" in str(result.output)


@pytest.mark.asyncio
async def test_exposure_policy_default_hides_write_and_terminal_tools(tmp_path):
    """contract #4 e2e: under the default Exposure Policy (expose_execution_tools=
    False) the mutating builtin tools never reach the model -- the assembled
    toolset carries only the read-only file tools."""
    from linktools.ai.agent.spec import PromptSpec
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.execution.local import LocalExecutionBackend
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.router import ModelRouter
    from linktools.ai.model.registry import ModelRegistry
    from pydantic_ai.models.function import FunctionModel

    registry = ModelRegistry()
    registry.register(
        "m",
        model=FunctionModel(lambda m, i: ModelResponse(parts=[TextPart(content="ok")])),
    )
    backend = LocalExecutionBackend(runtime_dir=tmp_path)
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        execution=backend,
        options=CapabilityRuntimeOptions(
            tool_exposure=CapabilityToolExposurePolicy()
        ),  # defaults: no execution tools
    )
    spec = AgentSpec(
        id="e2e",
        name="e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="*"),),  # builtin:* but mutating tools gated off
    )
    inspection = await rt.inspect(spec, execution=backend)
    names = {tool.name for tool in inspection.tools}
    assert {"list_dir", "read_file"} <= names
    assert "write_file" not in names
    assert "bash" not in names
