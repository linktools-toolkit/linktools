#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_compiler_tools.py — verifies the contract contract: the
AgentCompiler NO LONGER wires builtin file/terminal tools into the compiled
pydantic-ai Agent. Those tools are constructed at EXECUTION TIME from
``AgentDependencies.sandbox`` (set by AgentEngine from its ``sandbox``
kwarg) and passed via ``agent.iter(prompt, toolsets=[...])``.

Three angles:
1. A freshly-compiled Agent carries NO user-supplied FunctionToolsets (the
   builtin tools are not baked in at compile time). This replaces the old
   ``workdir=`` gate test.
2. A run driven by a runner WITHOUT an execution backend exposes no builtin
   tools -- a FunctionModel that tries to call read_file gets a tool-error
   back (pydantic-ai's "unknown tool" surface), never a file payload.
3. A run driven by a runner WITH a LocalSandbox wired sees a real
   read_file tool call land on the backend -- the file content shows up as
   a tool-return in the run history. This is the positive-path replacement
   for the old "compiled agent has builtin tools" test, now driven through
   the runner per contract's execution-time construction."""

import asyncio

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.sandbox.local import LocalSandbox
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker

from datetime import datetime, timezone


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


def _make_runner(tmp_path, *, sandbox=None) -> AgentEngine:
    from linktools.ai.capability.resolver import CapabilityResolver
    from linktools.ai.capability.builtin import BuiltinProvider
    from linktools.ai.governance.policy.engine import PolicyEngine
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
    from linktools.ai.tool.executor import GovernedToolInvoker

    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    return AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        checkpoint_store=checkpoint_store,
        sandbox=sandbox,
        capability_resolver=CapabilityResolver({"builtin": BuiltinProvider()}),
        managed_tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )


def _seed_session(store, session_id) -> None:
    now = datetime.now(timezone.utc)
    asyncio.run(
        store.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
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
    # When no Sandbox is wired, the runner-driven run exposes no
    # builtin tools. A FunctionModel that emits a read_file ToolCallPart
    # cannot land it on a backend. Drive the run via run_stream and collect
    # every yielded event: without a backend, NO successful "tool" event for
    # read_file surfaces (the model's tool call is rejected as unknown before
    # the builtin handler runs). The model then terminates with a final
    # response on its next turn.
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
    runner = _make_runner(tmp_path)  # sandbox=None -> no builtin tools
    _seed_session(runner._session_store, "session-1")

    async def _collect():
        events: "list[dict]" = []
        async for ev in runner.run_stream(
            compiled,
            RunInput(prompt="read sample.txt"),
            _run_context(),
        ):
            events.append(ev)
        return events

    events = asyncio.run(_collect())

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
    # Positive path: with a LocalSandbox wired into the runner, a
    # read_file tool call from the model lands on the backend. The runner
    # surfaces the call as a "tool" event via run_stream -- assert read_file
    # fires a successful "end" event AND the file content shows up in the
    # checkpointed message history (which the runner saves from
    # ``run.all_messages()`` -- this is where the tool-return payload lives).
    # This is the contract replacement for the old "compiled agent has builtin
    # tools" test, now driven through the runner per the execution-time
    # construction.
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
    runner = _make_runner(tmp_path, sandbox=backend)
    _seed_session(runner._session_store, "session-1")

    async def _drive():
        events: "list[dict]" = []
        async for ev in runner.run_stream(
            compiled,
            RunInput(prompt="read sample.txt"),
            _run_context(),
        ):
            events.append(ev)
        return events

    events = asyncio.run(_drive())

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

    # And the file content reached the message history -- the checkpoint
    # payload holds the serialized ``run.all_messages()`` with tool-returns.
    checkpoint = asyncio.run(runner._checkpoint_store.latest("run-1"))
    assert checkpoint is not None, "expected a checkpoint after the run"
    assert "hello from workdir" in str(checkpoint.payload), (
        "file content should appear in the checkpointed message history"
    )
