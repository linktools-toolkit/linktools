#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_compiler_tools.py — verifies the contract: the
AgentCompiler NO LONGER wires builtin file/terminal tools into the compiled
pydantic-ai Agent. Those tools are constructed at EXECUTION TIME from
``AgentDependencies.sandbox`` (set by AgentEngine from its ``sandbox``
kwarg) and passed via ``agent.iter(prompt, toolsets=[...])``.

Three angles:
1. A freshly-compiled Agent carries NO user-supplied FunctionToolsets (the
   builtin tools are not baked in at compile time). This replaces the old
   ``workdir=`` gate test.
2. A run driven by an engine WITHOUT an execution backend exposes no builtin
   tools -- a FunctionModel that tries to call read_file gets no successful
   tool event (the builtin handler never runs without a backend).
3. A run driven by an engine WITH a LocalSandbox wired sees a real read_file
   tool call land on the backend -- the file content shows up in the run's
   checkpoint payload (the serialized ``run.all_messages()`` execute_pure
   returns on AgentCompleted).

FS-29: the legacy ``run_stream`` is gone; cases 2-3 now drive ``execute_pure``
directly and read tool events from the injected ``live_events`` sink + the
checkpoint payload from the AgentCompleted outcome (the engine no longer holds
a commit_coordinator/checkpoint store)."""

import asyncio

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.models import AgentCompleted, AgentInput
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.sandbox.local import LocalSandbox
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.context import RunContext
from linktools.ai.run.live_events import NullSecurityEventSink
from linktools.ai.run.models import RunnableType
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


class _CollectingLiveEvents:
    """Captures every dict event execute_pure publishes via live_events."""

    def __init__(self) -> None:
        self.events: "list[dict]" = []

    async def publish(self, event: dict) -> None:
        self.events.append(event)


def _registry(model_fn) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _spec() -> AgentSpec:
    return AgentSpec(
        id="agent-tools",
        name="tools-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )


def _user_function_toolsets(compiled) -> "list[FunctionToolset]":
    """Return only user-supplied FunctionToolsets on the compiled Agent --
    pydantic-ai always carries its internal ``_AgentFunctionToolset`` for
    output-schema dispatch, so filter by exact class."""
    return [
        ts for ts in compiled.pydantic_agent.toolsets if type(ts) is FunctionToolset
    ]


def _make_runner(*, sandbox=None) -> AgentEngine:
    from linktools.ai.capability.resolver import CapabilityResolver
    from linktools.ai.capability.builtin import BuiltinProvider

    # FS-29: AgentEngine takes only its pure-execution dependencies -- no
    # run/session/event/checkpoint Store, no commit_coordinator. The
    # execution-time builtin-tool construction under test lives in
    # execute_pure's capability path (driven by AgentDependencies.sandbox).
    return AgentEngine(
        sandbox=sandbox,
        capability_resolver=CapabilityResolver({"builtin": BuiltinProvider()}),
        managed_tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
    )


def _run_context() -> RunContext:
    return RunContext(
        run_id="run-1",
        root_run_id="run-1",
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-tools",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _execute(runner, compiled, prompt) -> "tuple[AgentCompleted, list[dict]]":
    live = _CollectingLiveEvents()
    outcome = asyncio.run(
        runner.execute_pure(
            compiled,
            AgentInput(prompt=prompt),
            _run_context(),
            cancellation=CancellationToken(),
            live_events=live,
            security_events=NullSecurityEventSink(),
        )
    )
    return outcome, live.events


def test_compiled_agent_has_no_builtin_toolsets_at_compile_time():
    # contract: the compiler produces an Agent with NO builtin file/terminal tools.
    # Those tools are constructed at execution time, not compile time.
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(
            registry=_registry(
                lambda m, i: ModelResponse(parts=[TextPart(content="ok")])
            )
        ),
    )
    compiled = asyncio.run(compiler.compile(_spec()))

    assert _user_function_toolsets(compiled) == [], (
        "compiler must not bake builtin tools into the compiled Agent"
    )


def test_runner_without_execution_backend_exposes_no_builtin_tools(tmp_path):
    # When no Sandbox is wired, execute_pure exposes no builtin tools. A
    # FunctionModel that emits a read_file ToolCallPart cannot land it on a
    # backend. Without a backend, NO successful "tool" event for read_file
    # surfaces (the builtin handler never runs); the model then terminates
    # with a final response on its next turn.
    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        # Terminate on any non-first call. messages[0] is the user prompt;
        # the second call arrives after pydantic-ai has processed the prior
        # tool-call response (rejected as unknown -> retry-prompt to model).
        if len(messages) <= 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_file", args={"path": "sample.txt"}),
                ]
            )
        return ModelResponse(parts=[TextPart(content='{"response": {"done": true}}')])

    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(model_fn)),
    )
    compiled = asyncio.run(compiler.compile(_spec()))
    runner = _make_runner()  # sandbox=None -> no builtin tools

    _outcome, events = _execute(runner, compiled, "read sample.txt")

    # No successful read_file tool event -- the tool was unknown to the agent
    # (no execution backend -> no builtin tools registered on the iter() call).
    read_file_ok = [
        e
        for e in events
        if e.get("type") == "tool"
        and e.get("name") == "read_file"
        and e.get("phase") == "end"
        and e.get("ok")
    ]
    assert read_file_ok == [], (
        "no execution backend -> read_file must not produce a successful tool event"
    )


def test_runner_with_execution_backend_routes_read_file_to_backend(tmp_path):
    # Positive path: with a LocalSandbox wired into the engine, a read_file
    # tool call from the model lands on the backend. execute_pure surfaces the
    # call as a "tool" event via the live_events sink AND the file content
    # shows up in the checkpoint payload (the serialized run.all_messages()
    # carried on the AgentCompleted outcome -- where the tool-return payload
    # lives). This is the execution-time construction contract.
    (tmp_path / "sample.txt").write_text("hello from workdir", encoding="utf-8")

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        # Terminate after the first turn so the run completes cleanly
        # (pydantic-ai would otherwise loop on tool calls until its request
        # limit). The dict output schema requires a JSON object with a
        # `response` key, so the final turn emits that shape.
        if len(messages) <= 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="read_file", args={"path": "sample.txt"}),
                ]
            )
        return ModelResponse(
            parts=[TextPart(content='{"response": {"status": "done"}}')]
        )

    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=_registry(model_fn)),
    )
    compiled = asyncio.run(compiler.compile(_spec()))
    backend = LocalSandbox(runtime_dir=tmp_path)
    runner = _make_runner(sandbox=backend)

    outcome, events = _execute(runner, compiled, "read sample.txt")

    # read_file fired and completed successfully.
    read_file_ends = [
        e
        for e in events
        if e.get("type") == "tool"
        and e.get("name") == "read_file"
        and e.get("phase") == "end"
    ]
    assert read_file_ends, "expected read_file to have been called"
    assert all(e.get("ok") for e in read_file_ends), (
        f"read_file end events should be ok: {read_file_ends}"
    )

    # And the file content reached the message history -- the AgentCompleted
    # checkpoint payload holds the serialized ``run.all_messages()`` with
    # tool-returns.
    assert isinstance(outcome, AgentCompleted), "expected the run to complete"
    assert "hello from workdir" in str(outcome.checkpoint_payload), (
        "file content should appear in the checkpointed message history"
    )
