"""Full Runtime-to-MCP execution contract."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.events.payloads import (
    ToolCompleted,
    ToolPipelineDecision,
    ToolPolicyResolved,
    ToolStarted,
)
from linktools.ai.mcp.client import MCPConnectionRef
from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPProvider, MCPToolInfo
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.policy.engine import PolicyEngine
from linktools.ai.security.baseline import SecurityBaseline
from linktools.ai.security.pipeline import PipelineAction, PipelineDecision
from linktools.ai.providers.bundle import ProviderBundle
from linktools.ai.registry.mcp import MCPServerSpec
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage
from linktools.ai.tool.executor import ToolExecutor


class _CountingPolicy(PolicyEngine):
    def __init__(self):
        super().__init__(rules=())
        self.count = 0

    async def evaluate(self, request, context):
        self.count += 1
        return await super().evaluate(request, context)


class _CountingExecutor:
    def __init__(self, delegate):
        self.delegate = delegate
        self.count = 0

    async def is_approved(self, run_id, call_id):
        return await self.delegate.is_approved(run_id, call_id)

    async def execute(self, *args, **kwargs):
        self.count += 1
        return await self.delegate.execute(*args, **kwargs)


class _CountingPipeline:
    def __init__(self):
        self.before = 0
        self.after = 0

    async def before_tool(self, event):
        self.before += 1
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def after_tool(self, event):
        self.after += 1
        return PipelineDecision(action=PipelineAction.ALLOW)


class _SpecProvider:
    def __init__(self, *, enabled_tools=None):
        self.enabled_tools = enabled_tools

    async def list_ids(self):
        return ("demo",)

    async def get(self, server_id):
        return MCPServerSpec(
            id=server_id,
            name=server_id,
            transport="stdio",
            command=("demo",),
            enabled_tools=self.enabled_tools,
        )


class _Manager:
    def __init__(self):
        self.ref = MCPConnectionRef("demo", "fingerprint")
        self.calls = []

    async def list_tools_result(self, server):
        return MCPDiscoveryResult(
            tools=(
                MCPToolInfo(
                    name="lookup",
                    description="Look up a value",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    read_only=True,
                ),
            ),
            verified=True,
            connection_ref=self.ref,
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        self.calls.append((connection_ref, tool_name, arguments))
        return {"value": "mcp-value"}


def _router():
    def model_fn(messages, info: AgentInfo):
        if not any(
            getattr(part, "part_kind", None) == "tool-return"
            for message in messages
            for part in message.parts
        ):
            return ModelResponse(
                parts=[ToolCallPart(tool_name="demo.lookup", args={"key": "x"})]
            )
        return ModelResponse(parts=[TextPart(content="mcp-value")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return ModelRouter(registry=registry)


def _text_router():
    def model_fn(messages, info: AgentInfo):
        return ModelResponse(parts=[TextPart(content="no tools needed")])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return ModelRouter(registry=registry)


@pytest.mark.asyncio
async def test_runtime_runs_mcp_through_managed_execution(tmp_path):
    manager = _Manager()
    provider = MCPProvider(_SpecProvider(), manager)
    runtime = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=_router(),
        providers=ProviderBundle(capabilities=(provider,)),
    )
    spec = AgentSpec(
        id="agent",
        name="agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="use the lookup tool"),
        tools=(ToolRef(kind="mcp", name="demo"),),
    )

    inspection = await runtime.inspect(spec)
    assert [tool.name for tool in inspection.tools] == ["demo.lookup"]
    assert not any(hasattr(tool, "handler") for tool in inspection.tools)

    result = await runtime.run(spec, "look up x")
    assert "mcp-value" in str(getattr(result, "output", result))
    assert manager.calls == [(manager.ref, "lookup", {"key": "x"})]


@pytest.mark.asyncio
async def test_runtime_mcp_governance_executes_exactly_once(tmp_path):
    manager = _Manager()
    provider = MCPProvider(_SpecProvider(), manager)
    storage = FileStorage(root=tmp_path)
    policy = _CountingPolicy()
    executor = _CountingExecutor(ToolExecutor(policy=policy))
    pipeline = _CountingPipeline()
    runtime = Runtime.build(
        storage=storage,
        model_router=_router(),
        tool_executor=executor,
        security=SecurityBaseline(pipeline=pipeline),
        providers=ProviderBundle(capabilities=(provider,)),
    )
    spec = AgentSpec(
        id="agent",
        name="agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="use the lookup tool"),
        tools=(ToolRef(kind="mcp", name="demo"),),
    )

    await runtime.run(spec, "look up x", run_id="run-e2e")

    assert executor.count == 1
    assert policy.count == 1
    assert pipeline.before == 1
    assert pipeline.after == 1
    assert len(manager.calls) == 1
    events = (await storage.events.list("run-e2e", limit=100)).items
    payloads = [event.payload for event in events]
    assert sum(isinstance(event, ToolPolicyResolved) for event in payloads) == 1
    assert (
        sum(
            isinstance(event, ToolPipelineDecision) and event.stage == "before"
            for event in payloads
        )
        == 1
    )
    assert (
        sum(
            isinstance(event, ToolPipelineDecision) and event.stage == "after"
            for event in payloads
        )
        == 1
    )
    assert sum(isinstance(event, ToolStarted) for event in payloads) == 1
    assert sum(isinstance(event, ToolCompleted) for event in payloads) == 1


@pytest.mark.asyncio
async def test_runtime_empty_mcp_allowlist_exposes_and_calls_no_tools(tmp_path):
    manager = _Manager()
    provider = MCPProvider(_SpecProvider(enabled_tools=()), manager)
    runtime = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=_text_router(),
        providers=ProviderBundle(capabilities=(provider,)),
    )
    spec = AgentSpec(
        id="agent",
        name="agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="do not use tools"),
        tools=(ToolRef(kind="mcp", name="demo"),),
    )

    inspection = await runtime.inspect(spec)
    assert inspection.tools == ()
    result = await runtime.run(spec, "answer without tools")
    assert "no tools needed" in str(getattr(result, "output", result))
    assert manager.calls == []
