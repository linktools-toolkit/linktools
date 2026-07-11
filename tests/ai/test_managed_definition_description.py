import pytest
from types import SimpleNamespace

from linktools.ai.capability import CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.mcp.client import MCPConnectionManager
from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPProvider, MCPToolInfo
from linktools.ai.providers.mcp import MCPServerSpecProvider
from linktools.ai.registry.mcp import parse_mcp_spec
from linktools.ai.security.descriptor import ToolDescriptor
from linktools.ai.tool.contribution import ManagedToolDefinition


def _descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="t", source="test", category="c", risk="low", mutating=False)


async def _handler(**arguments):
    return arguments


def test_managed_tool_definition_has_optional_description():
    md = ManagedToolDefinition(descriptor=_descriptor(), handler=_handler)
    assert md.description is None
    md2 = ManagedToolDefinition(
        descriptor=_descriptor(), handler=_handler, description="real desc")
    assert md2.description == "real desc"


def test_toolset_uses_definition_description_when_present():
    from linktools.ai.agent.runner import _toolset_for_definition

    md = ManagedToolDefinition(
        descriptor=_descriptor(), handler=_handler,
        parameters_json_schema={"type": "object", "properties": {}},
        description="A tool that does a thing",
    )
    ts = _toolset_for_definition(md)
    tool = ts.tools["t"]
    assert getattr(tool, "description", None) == "A tool that does a thing"


def test_toolset_falls_back_to_descriptor_name_without_description():
    from linktools.ai.agent.runner import _toolset_for_definition

    md = ManagedToolDefinition(
        descriptor=_descriptor(), handler=_handler,
        parameters_json_schema={"type": "object", "properties": {}},
        description=None,
    )
    ts = _toolset_for_definition(md)
    tool = ts.tools["t"]
    # Falls back to the descriptor name so the model always sees a description.
    assert getattr(tool, "description", None) == "t"


# --- MCPProvider description + read_only flow ---


class _InfoSpecProvider(MCPServerSpecProvider):
    def __init__(self, spec):
        self._spec = spec

    async def list_ids(self):
        return ("risk",)

    async def get(self, server_id):
        return self._spec


class _InfoManager:
    """Returns MCPToolInfo entries (with description + read_only) through the
    verified discovery path the real MCPConnectionManager exposes."""

    def __init__(self, infos):
        self._infos = tuple(infos)

    async def list_tools(self, spec):
        return tuple(i.name for i in self._infos)

    async def get_toolset(self, spec):
        from pydantic_ai.toolsets import FunctionToolset
        return FunctionToolset()

    async def list_tools_result(self, spec):
        return MCPDiscoveryResult(tools=self._infos, verified=True, connection_ref=None)

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        raise RuntimeError("not invoked during resolution")


def _resolve_bundle(manager, spec):
    provider = MCPProvider(_InfoSpecProvider(spec), manager)
    import asyncio

    async def run():
        return await provider.resolve(
            CapabilityRef("mcp", "risk"),
            CapabilityContext(agent_id="a1", exposure_policy=CapabilityToolExposurePolicy()),
        )

    return asyncio.run(run())


def test_mcp_description_flows_into_managed_definition():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]})
    info = MCPToolInfo(
        name="query_user",
        parameters_json_schema={"type": "object", "properties": {}},
        description="Look up a user by id",
        read_only=True,
    )
    bundle = _resolve_bundle(_InfoManager([info]), spec)
    definition = bundle.tool_contributions[0].tools[0]
    assert definition.description == "Look up a user by id"
    assert definition.descriptor.name == "risk.query_user"


def test_mcp_read_only_hint_marks_descriptor_non_mutating():
    spec = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]})

    ro = MCPToolInfo(name="get_user",
                     parameters_json_schema={"type": "object"}, read_only=True)
    rw = MCPToolInfo(name="set_user",
                     parameters_json_schema={"type": "object"}, read_only=False)
    unknown = MCPToolInfo(name="mystery",
                          parameters_json_schema={"type": "object"})

    bundle = _resolve_bundle(_InfoManager([ro, rw, unknown]), spec)
    by_name = {md.descriptor.name: md for md in bundle.tool_contributions[0].tools}

    assert by_name["risk.get_user"].descriptor.mutating is False
    assert by_name["risk.set_user"].descriptor.mutating is True
    # Unknown read-only status stays high-risk / mutating -- never inferred.
    assert by_name["risk.mystery"].descriptor.mutating is True


# --- Live discovery bridge: tool annotations -> MCPToolInfo.read_only ---


def _live_tool(*, name="t", read_only_hint="__absent__"):
    annotations = (None if read_only_hint == "__absent__"
                   else SimpleNamespace(readOnlyHint=read_only_hint))
    return SimpleNamespace(
        name=name, description="d",
        inputSchema={"type": "object", "properties": {}},
        annotations=annotations, metadata={})


def test_convert_tool_info_reads_read_only_hint_true():
    info = MCPConnectionManager._convert_tool_info(_live_tool(read_only_hint=True))
    assert info.read_only is True


def test_convert_tool_info_reads_read_only_hint_false():
    info = MCPConnectionManager._convert_tool_info(_live_tool(read_only_hint=False))
    assert info.read_only is False


def test_convert_tool_info_treats_absent_annotations_as_unknown():
    # No annotations at all -> read_only stays None (unknown), which the provider
    # maps to mutating/high-risk. A read-only-looking name never auto-qualifies.
    info = MCPConnectionManager._convert_tool_info(_live_tool())
    assert info.read_only is None
