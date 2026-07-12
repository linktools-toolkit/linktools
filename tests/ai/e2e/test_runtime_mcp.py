"""Full Runtime-to-MCP execution contract."""

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.mcp.client import MCPConnectionRef
from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPProvider, MCPToolInfo
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.router import ModelRouter
from linktools.ai.providers.bundle import ProviderBundle
from linktools.ai.registry.mcp import MCPServerSpec
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage


class _SpecProvider:
    async def list_ids(self):
        return ("demo",)

    async def get(self, server_id):
        return MCPServerSpec(
            id=server_id,
            name=server_id,
            transport="stdio",
            command=("demo",),
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
