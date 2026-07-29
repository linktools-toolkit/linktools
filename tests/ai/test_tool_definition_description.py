from types import SimpleNamespace

from linktools.ai.agent.assembly import AgentFeatureRef
from linktools.ai.agent.integrations.mcp.client import MCPClient
from linktools.ai.agent.tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from linktools.ai.governance.policy.rule import RiskLevel, SideEffectKind


async def _handler() -> str:
    return "ok"


def test_tool_definition_description_is_optional() -> None:
    definition = ToolDefinition(
        descriptor=ToolDescriptor(
            name="tool",
            source=ToolSource.EXTENSION,
            category=ToolCategory.EXTENSION_READ,
            risk=RiskLevel.LOW,
            side_effect=SideEffectKind.READ_ONLY,
            feature=AgentFeatureRef("extension", "test"),
        ),
        handler=_handler,
    )
    assert definition.description is None


def test_mcp_conversion_preserves_description_and_read_only_hint() -> None:
    live = SimpleNamespace(
        name="lookup",
        description="Look up an item",
        inputSchema={"type": "object", "properties": {}},
        annotations=SimpleNamespace(readOnlyHint=True),
        metadata={},
    )
    info = MCPClient.convert_tool_info(live)
    assert info.description == "Look up an item"
    assert info.read_only is True
