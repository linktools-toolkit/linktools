#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent declarations execute through the native SDK lifecycle."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import AgentContext, CapabilityContribution, ToolCallRetry, ToolResultsCapability
from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode


def _context(identity: str) -> AgentContext:
    return AgentContext(None, service_principal("tenant", "reader"), "example", None, identity, {})


def _native(capability, identity):
    return CapabilityContribution.from_opaque("capability", identity, capability, revision=2).materialize_capability()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("direct", "wrapper", "combined"))
async def test_native_projections_use_initialized_run_instances_in_sdk_order(shape):
    events = []

    class Project(ToolResultsCapability):
        def __init__(self, name, run=None):
            self.id, self.run, self.ready = name, run, False

        async def for_run(self, ctx):
            assert self.run is None
            bound = type(self)(self.id, ctx.deps.execution_id)
            events.append(("bind", self.id, bound.run, id(bound)))
            return bound

        async def before_run(self, ctx):
            self.ready = True

        async def wrap_run(self, ctx, *, handler):
            try:
                return await handler()
            finally:
                events.append(("closed", self.id, self.run, id(self)))
                self.ready = False

        async def process_tool_result(self, context, tool, arguments, result):
            assert self.ready and self.run == context.execution_id
            events.append(("project", self.id, self.run, id(self)))
            await asyncio.sleep(0)
            return result + ":" + self.id

        async def after_tool_execute(self, ctx, *, result, **kwargs):
            events.append(("after", self.id, self.run, id(self)))
            return result

    caps = []
    for name in ("first", "second"):
        cap = Project(name)
        if shape == "wrapper":
            cap = WrapperCapability(cap)
        elif shape == "combined":
            cap = CombinedCapability([cap])
        caps.append(_native(cap, name))

    async def inspect() -> str:
        return "raw"

    def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="same-call")])
        assert returns[-1].content == "raw:first:second"
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(model=FunctionModel(respond), tools=[inspect], capabilities=caps)
    results = await asyncio.gather(*(agent.run("Inspect", deps=_context(run)) for run in ("A", "B")))
    assert [result.output for result in results] == ["done", "done"]
    for run in ("A", "B"):
        for name in ("first", "second"):
            subset = [event for event in events if event[1:3] == (name, run)]
            assert {event[0] for event in subset} == {"bind", "project", "after", "closed"}
            assert len({event[3] for event in subset}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_error", (ValueError, ToolCallRetry, asyncio.CancelledError))
async def test_native_projection_failure_never_retries_a_completed_tool(projection_error):
    calls, requests, cleaned = 0, 0, []

    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            try:
                raise projection_error("projection failed")
            finally:
                cleaned.append(True)

    async def perform() -> str:
        nonlocal calls
        calls += 1
        return "performed"

    def respond(messages, info):
        nonlocal requests
        requests += 1
        return ModelResponse(parts=[ToolCallPart("perform", {}, tool_call_id="perform")])

    agent = Agent(model=FunctionModel(respond), tools=[perform], capabilities=[_native(Project(), "project")])
    expected = asyncio.CancelledError if projection_error is asyncio.CancelledError else AIError
    with pytest.raises(expected) as caught:
        await agent.run("Perform", deps=_context("run"))
    if expected is AIError:
        assert caught.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    assert calls == requests == 1
    assert cleaned == [True]


@pytest.mark.asyncio
async def test_native_error_recovery_is_not_a_raw_success_projection():
    projected = []

    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            projected.append(result)
            return "projected"

    class Recover(AbstractCapability):
        async def on_tool_execute_error(self, ctx, *, error, **kwargs):
            assert isinstance(error, ValueError)
            return "recovered"

    async def fail() -> str:
        raise ValueError("raw failure")

    def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("fail", {}, tool_call_id="fail")])
        assert returns[-1].content == "recovered"
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(model=FunctionModel(respond), tools=[fail], capabilities=[_native(Project(), "project"), Recover()])
    assert (await agent.run("Fail", deps=_context("run"))).output == "done"
    assert projected == []


@pytest.mark.asyncio
async def test_bound_workspace_enforces_its_policy_without_a_runtime(tmp_path):
    from pydantic_ai import DeferredToolRequests, DeferredToolResults
    from linktools.ai.capability import PydanticToolControlCapability, FileSystem, ShellCapability

    from linktools.ai.workspace import (
        LocalWorkspaceBackend, ToolPermissionPolicy, ToolPermissionRule, Workspace, WorkspacePolicy,
    )

    workspace = Workspace(tmp_path, {}, WorkspacePolicy(tool_permissions=ToolPermissionPolicy(
        rules=(ToolPermissionRule("ask", tool_class="filesystem.write"),
               ToolPermissionRule("deny", tool_class="shell")),
    )))
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    calls = [
        ToolCallPart("write_file", {"path": "approved.txt", "content": "approved"}, tool_call_id="write"),
        ToolCallPart("run_command", {"command": "printf forbidden > denied.txt"}, tool_call_id="shell"),
        ToolCallPart("business_echo", {}, tool_call_id="business"),
    ]

    def business_echo() -> str:
        return "unrelated"

    def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            if returns[-1].tool_call_id == "shell":
                assert returns[-1].outcome == "failed"
            if returns[-1].tool_call_id == "business":
                assert returns[-1].content == "unrelated"
        if len(returns) < len(calls):
            return ModelResponse(parts=[calls[len(returns)]])
        return ModelResponse(parts=[TextPart("done")])

    try:
        cap = FileSystem(workspace).bind(("write_file",), backend=session)
        shell_cap = ShellCapability().bind(("run_command",), backend=session, policy=workspace.policy.tool_permissions)
        backend = session
        agent = Agent(model=FunctionModel(respond), tools=[business_echo],
                      capabilities=[PydanticToolControlCapability(), cap, shell_cap],
                      output_type=[str, DeferredToolRequests])
        first = await agent.run("Write, then try shell", deps=_context("first"), workspace=backend)
        assert isinstance(first.output, DeferredToolRequests)
        assert not (tmp_path / "approved.txt").exists()
        second = await agent.run(message_history=first.all_messages(),
                                 deferred_tool_results=DeferredToolResults(approvals={"write": True}),
                                 deps=_context("resumed"), workspace=backend)
        assert second.output == "done"
        assert (tmp_path / "approved.txt").read_text() == "approved"
        assert not (tmp_path / "denied.txt").exists()
        assert "approved.txt" in await session.list_directory(".")
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_native_references_use_actual_loader_and_only_ephemeral_provider_messages(stream):
    import hashlib
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.capability import ReferencedInputsCapability
    from linktools.ai.model import ReferencedContent
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision

    body, events = b"authorized provider body", []
    content = ReferencedContent(
        AssetVersionRef(AssetKey("document", "one"), "group", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type="text/plain", representation="text", source=CapabilityRef("capability", "source", 2),
    )

    class Source(ReferencedInputsCapability):
        id = "source"

        def __init__(self, run=None):
            self.run, self.ready = run, False
            async def load(context, reference):
                assert self.ready and self.run == context.execution_id
                assert reference is content
                events.append(("load", self.run, id(self)))
                return body
            super().__init__(load)

        async def for_run(self, ctx):
            bound = type(self)(ctx.deps.execution_id)
            events.append(("bind", bound.run, id(bound)))
            return bound

        async def before_run(self, ctx):
            self.ready = True

        async def wrap_run(self, ctx, *, handler):
            try:
                return await handler()
            finally:
                self.ready = False
                events.append(("closed", self.run, id(self)))

    def observe(messages):
        from pydantic_ai.messages import UserPromptPart
        assert any(body.decode() in part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart) and not isinstance(part.content, str))

    def respond(messages, info):
        observe(messages)
        return ModelResponse(parts=[TextPart("done")])

    async def respond_stream(messages, info):
        observe(messages)
        yield "done"

    class InputModel(FunctionModel):
        def validate_inputs(self, value):
            # This synthetic provider explicitly owns its text input contract.
            assert isinstance(value, list)

    agent = Agent(model=InputModel(respond, stream_function=respond_stream),
                  capabilities=[_native(Source(), "source")])
    async def run(identity):
        if stream:
            async with agent.run_stream([content], deps=_context(identity)) as result:
                assert await result.get_output() == "done"
                messages = result.all_messages()
        else:
            result = await agent.run([content], deps=_context(identity))
            assert result.output == "done"
            messages = result.all_messages()
        assert body.decode() not in repr(messages)

    await asyncio.gather(run("A"), run("B"))
    for identity in ("A", "B"):
        local = [event for event in events if event[1] == identity]
        assert [event[0] for event in local] == ["bind", "load", "closed"]
        assert len({event[2] for event in local}) == 1


@pytest.mark.asyncio
async def test_native_reference_rejects_unknown_actual_model_before_loading():
    import hashlib
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.capability import ReferencedInputsCapability
    from linktools.ai.model import ReferencedContent
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision

    loads, calls = [], []
    body = b"protected"
    content = ReferencedContent(
        AssetVersionRef(AssetKey("document", "one"), "group", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type="text/plain", representation="text", source=CapabilityRef("capability", "source", 2),
    )
    async def load(context, reference):
        loads.append(reference)
        return body
    def respond(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])
    class TrustedModel(FunctionModel):
        def validate_inputs(self, value):
            pass
    agent = Agent(model=TrustedModel(respond),
                  capabilities=[_native(ReferencedInputsCapability(load), "source")])
    with pytest.raises(AIError) as caught:
        await agent.run([content], model=FunctionModel(respond), deps=_context("run"))
    assert caught.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert loads == calls == []


@pytest.mark.parametrize("mode", ("session", "execution", "runtime-scope"))
def test_native_declarations_do_not_silently_supply_host_services(mode):
    from linktools.ai.capability import ExecutionCapability, SessionCapability
    if mode == "runtime-scope":
        contribution = CapabilityContribution.from_opaque(
            "capability", "plain", AbstractCapability(), scope="runtime",
        )
        with pytest.raises(AIError) as caught:
            contribution.materialize_capability()
        assert caught.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
        return
    capability = SessionCapability() if mode == "session" else ExecutionCapability()
    with pytest.raises(AIError):
        Agent(model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("unexpected")])),
              capabilities=[capability])


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("direct", "combined", "wrapper"))
async def test_native_deferred_projection_obeys_sdk_activation_and_replaced_run_shape(shape):
    events = []
    def container(capability):
        if shape == "combined":
            return CombinedCapability([capability])
        return WrapperCapability(capability) if shape == "wrapper" else capability

    class Project(ToolResultsCapability):
        id, defer_loading = "project", True
        def __init__(self, bound=False):
            self.bound = bound
        async def for_run(self, ctx):
            assert not self.bound
            events.append("bind")
            return container(type(self)(True))
        async def process_tool_result(self, context, tool, arguments, result):
            assert self.bound
            events.append(tool.name)
            return "projected:" + result

    requests = 0
    def respond(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="before")])
        if requests == 2:
            assert events == ["bind"]
            assert messages[-1].parts[0].content == "raw"
            return ModelResponse(parts=[ToolCallPart("load_capability", {"id": "project"}, tool_call_id="load")])
        if requests == 3:
            return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="after")])
        assert requests == 4 and messages[-1].parts[0].content == "projected:raw"
        return ModelResponse(parts=[TextPart("done")])
    async def inspect() -> str:
        return "raw"
    agent = Agent(model=FunctionModel(respond), tools=[inspect], capabilities=[_native(Project(), "project")])
    assert (await agent.run("Inspect", deps=_context("run"))).output == "done"
    assert events == ["bind", "inspect"]


@pytest.mark.asyncio
async def test_native_mcp_and_function_tools_share_one_projection_boundary(tmp_path):
    import sys
    from fastmcp.client.transports import StdioTransport
    from pydantic_ai.mcp import MCPToolset

    server = tmp_path / "server.py"
    server.write_text('''import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "example", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "inspect", "description": "Inspect synthetic data", "inputSchema": {"type": "object", "properties": {}}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "mcp-result"}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
''')
    processed = []
    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            processed.append((tool.name, result))
            return "projected:" + str(result)
    def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="mcp")])
        assert returns[-1].content == "projected:mcp-result"
        return ModelResponse(parts=[TextPart("done")])
    agent = Agent(model=FunctionModel(respond), capabilities=[_native(Project(), "project")],
                  toolsets=[MCPToolset(StdioTransport(sys.executable, args=[str(server)]))])
    async with agent:
        assert (await agent.run("Inspect", deps=_context("mcp-run"))).output == "done"
    assert processed == [("inspect", "mcp-result")]


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_model", (False, True))
async def test_native_reference_provider_view_precedes_compaction_and_follows_model_selection(replace_model):
    import hashlib
    from pydantic_ai.usage import RequestUsage, UsageLimits
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.capability import ReferencedInputsCapability
    from linktools.ai.model import ReferencedContent
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision

    body, calls, loaded = b"only the current provider sees this", [], []
    content = ReferencedContent(
        AssetVersionRef(AssetKey("document", "one"), "group", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type="text/plain", representation="text", source=CapabilityRef("capability", "source", 2),
    )
    async def load(context, reference):
        loaded.append(reference)
        return body

    class InputModel(FunctionModel):
        def __init__(self, name):
            self.identity = name
            def respond(messages, info):
                self.observe("request", messages)
                return ModelResponse(parts=[TextPart("done")])
            super().__init__(respond)
        def observe(self, operation, messages):
            from pydantic_ai.messages import UserPromptPart
            assert any(body.decode() in part.content for message in messages for part in message.parts
                       if isinstance(part, UserPromptPart) and not isinstance(part.content, str))
            calls.append((self.identity, operation))
        def validate_inputs(self, value):
            pass
        async def count_tokens(self, messages, model_settings, model_request_parameters):
            self.observe("count", messages)
            return RequestUsage(input_tokens=1)
        async def compact_messages(self, request_context, *, instructions=None):
            self.observe("compact", request_context.messages)
            return ModelResponse(parts=[TextPart("compacted")])

    first, selected = InputModel("first"), InputModel("selected")
    class Compact(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            await request_context.model.compact_messages(request_context)
            return request_context
    class Replace(AbstractCapability):
        async def before_model_request(self, ctx, request_context):
            request_context.model = selected
            return request_context

    capabilities = [_native(ReferencedInputsCapability(load), "source")]
    capabilities.append(Replace() if replace_model else Compact())
    agent = Agent(model=first, capabilities=capabilities)
    result = await agent.run([content], deps=_context("run"),
                             usage_limits=UsageLimits(count_tokens_before_request=True))
    assert result.output == "done"
    assert calls == ([("selected", "count"), ("selected", "request")] if replace_model else
                     [("first", "compact"), ("first", "count"), ("first", "request")])
    assert len(loaded) == len(calls)
    assert body.decode() not in repr(result.all_messages())


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ("direct", "wrapper", "combined"))
@pytest.mark.parametrize("phase", ("before", "wrap"))
async def test_final_input_guard_cannot_be_bypassed_by_another_innermost_capability(shape, phase):
    import hashlib
    from pydantic_ai.capabilities import CapabilityOrdering
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.capability import ReferencedInputsCapability
    from linktools.ai.model import ReferencedContent
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision
    body, events = b"protected", []
    content = ReferencedContent(
        AssetVersionRef(AssetKey("document", "one"), "group", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type="text/plain", representation="text", source=CapabilityRef("capability", "source", 2),
    )
    async def load(context, reference):
        events.append("load")
        return body
    def respond(messages, info):
        events.append("provider")
        return ModelResponse(parts=[TextPart("unexpected")])
    class InitialModel(FunctionModel):
        def validate_inputs(self, value):
            pass
    selected = FunctionModel(respond)
    class LateReplace(AbstractCapability):
        def get_ordering(self):
            return CapabilityOrdering(position="innermost")
        async def before_model_request(self, ctx, request_context):
            if phase == "before":
                request_context.model = selected
            return request_context
        async def wrap_model_request(self, ctx, *, request_context, handler):
            if phase == "wrap":
                request_context.model = selected
            return await handler(request_context)
    replacement = LateReplace()
    if shape == "wrapper":
        replacement = WrapperCapability(replacement)
    elif shape == "combined":
        replacement = CombinedCapability([replacement])
    agent = Agent(model=InitialModel(respond), capabilities=[
        _native(ReferencedInputsCapability(load), "source"), replacement,
    ])
    with pytest.raises(AIError) as caught:
        await agent.run([content], deps=_context("run"))
    assert caught.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert events == []


@pytest.mark.asyncio
async def test_skill_resources_and_delegated_child_execute_with_independent_native_models():
    from linktools.ai.capability import (
        PydanticToolControlCapability, SkillCapability, SkillDefinition, SkillLocation,
        SkillSourceRef, SkillSourceRegistry, SkillSourceView, SubagentCapability,
    )
    from linktools.ai.spec import SkillSpec, SubagentRef

    class Source:
        source_id = "memory"
        closed = 0
        async def inspect(self, source):
            return SkillSourceView(SkillLocation("virtual", "guide"), ("guide.txt",))
        async def read(self, source, path):
            assert path == "guide.txt"
            return b"resource content"
        async def close(self):
            self.closed += 1
    source, delegated, child_calls = Source(), [], []
    async def child_tool(text: str) -> str:
        child_calls.append(text)
        return text.upper()
    def child_model(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("child_tool", {"text": "inspect"}, tool_call_id="child")])
        assert returns[-1].content == "INSPECT"
        return ModelResponse(parts=[TextPart("child result")])
    child = Agent(model=FunctionModel(child_model), tools=[child_tool])
    async def delegate(ref, task, *, files, invocation_id):
        delegated.append((ref.id, task, files, invocation_id))
        result = await child.run(task)
        return {"output": result.output}
    skill = SkillCapability([
        SkillDefinition(SkillSpec("guide", "actual skill instructions", "Guide"), SkillSourceRef("memory", "guide")),
    ], SkillSourceRegistry([source]))
    subagent = SubagentCapability([SubagentRef("agent", "child")], delegate)
    calls = [
        ToolCallPart("load_skill", {"skill_id": "guide"}, tool_call_id="root"),
        ToolCallPart("load_skill", {"skill_id": "guide", "path": "guide.txt"}, tool_call_id="resource"),
        ToolCallPart("delegate_task", {"subagent_id": "child", "task": " inspect "}, tool_call_id="delegate"),
    ]
    def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            expected = ("actual skill instructions", "resource content", "child result")[len(returns) - 1]
            assert expected in str(returns[-1].content)
        if len(returns) < len(calls):
            return ModelResponse(parts=[calls[len(returns)]])
        return ModelResponse(parts=[TextPart("done")])
    async with Agent(model=FunctionModel(respond),
                     capabilities=[PydanticToolControlCapability(), skill, subagent]) as agent:
        assert (await agent.run("Read and delegate")).output == "done"
    assert delegated == [("child", "inspect", (), "delegate")]
    assert child_calls == ["inspect"]
    assert source.closed == 0
    await source.close()
    assert source.closed == 1


@pytest.mark.asyncio
async def test_native_projection_cancellation_drains_the_owned_resource(tmp_path):
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.workspace import LocalWorkspaceBackend

    entered, cleaned, sessions = asyncio.Event(), [], []
    class Owned(AbstractCapability):
        def __init__(self, run=None):
            self.run, self.session = run, None
        async def for_run(self, ctx):
            return type(self)(ctx.deps.execution_id)
        async def wrap_run(self, ctx, *, handler):
            self.session = LocalWorkspaceBackend(tmp_path)
            sessions.append(self.session)
            try:
                return await handler()
            finally:
                await self.session.close()
                cleaned.append(self.run)
        def get_toolset(self):
            toolset = FunctionToolset()
            @toolset.tool_plain
            async def inspect() -> str:
                return await self.session.list_directory(".")
            return toolset
    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            entered.set()
            await asyncio.sleep(3600)
            return result
    requests = []
    def respond(messages, info):
        requests.append(messages)
        if len(requests) == 1:
            return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="inspect")])
        return ModelResponse(parts=[TextPart("done")])
    agent = Agent(model=FunctionModel(respond), capabilities=[
        WrapperCapability(Owned()), _native(Project(), "project"),
    ])
    task = asyncio.create_task(agent.run("Inspect", deps=_context("run")))
    waiting = asyncio.create_task(entered.wait())
    try:
        done, _ = await asyncio.wait((task, waiting), timeout=5, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            outcome = await task
            pytest.fail(str(outcome.all_messages()))
        assert waiting in done
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        waiting.cancel()
        await asyncio.gather(task, waiting, return_exceptions=True)
    assert cleaned == ["run"]
    with pytest.raises(AIError) as caught:
        await sessions[0].list_directory(".")
    assert caught.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
