import pytest

from linktools.ai.errors import (
    IdempotencyConfigurationError,
    MCPErrorCode,
)
from linktools.ai.mcp.provider import MCPExposedTool
from linktools.ai.mcp.client import (
    MCPConnectionManager,
    MCPConnectionRef,
    MCPToolsetHandle,
)
from linktools.ai.registry.mcp import MCPServerSpec
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.security.redact import redact_for_audit, redact_exception
from linktools.ai.security.emitter import DefaultSecurityEventSanitizer
from linktools.ai.tool.idempotency import DefaultIdempotencyKeyBuilder
from linktools.ai.tool.idempotency import encode_business_key
from linktools.ai.tool.policy import (
    EffectiveToolPolicy,
    IdempotencyStrategy,
    ResolvedToolPolicy,
)


def test_public_snapshots_freeze_nested_values_and_copy_inputs():
    metadata = {"nested": {"items": [1]}}
    descriptor = ToolDescriptor(
        name="tool",
        source="test",
        category="discovery",
        risk="low",
        mutating=False,
        metadata=metadata,
    )
    metadata["nested"]["items"].append(2)
    assert tuple(descriptor.metadata["nested"]["items"]) == (1,)
    with pytest.raises(TypeError):
        descriptor.metadata["nested"]["items"] = ()


def test_business_key_requires_trusted_configured_field():
    descriptor = ToolDescriptor(
        name="create",
        source="test",
        category="mcp-write",
        risk="high",
        mutating=True,
    )
    policy = EffectiveToolPolicy(
        idempotent=True,
        idempotency_strategy=IdempotencyStrategy.BUSINESS_KEY,
        idempotency_key_field="external_id",
    )
    with pytest.raises(IdempotencyConfigurationError):
        DefaultIdempotencyKeyBuilder().build(
            descriptor=descriptor,
            arguments={},
            run_context=type("C", (), {"run_id": "r"})(),
            schema_version="1",
            policy=policy,
        )


def test_redaction_recurses_and_masks_exception_tokens():
    value = redact_for_audit({"nested": [{"authorization": "Bearer abc"}]})
    assert value["nested"][0]["authorization"] == "***REDACTED***"
    assert "Bearer abc" not in redact_exception(RuntimeError("Bearer abc"))


def test_mcp_exposed_tool_is_deeply_immutable():
    tool = MCPExposedTool("server", "raw", "server.raw", metadata={"x": {"y": 1}})
    with pytest.raises(TypeError):
        tool.metadata["x"]["y"] = 2


def test_policy_strategy_is_normalized_at_the_domain_boundary():
    policy = ResolvedToolPolicy(
        idempotent=True,
        idempotency_strategy="business_key",
        idempotency_key_field="external_id",
    )
    assert policy.idempotency_strategy is IdempotencyStrategy.BUSINESS_KEY


@pytest.mark.parametrize("value", [True, {}, []])
def test_business_key_rejects_unstable_values(value):
    with pytest.raises(IdempotencyConfigurationError):
        encode_business_key(value)


def test_mcp_error_codes_are_stable_and_sanitizer_bounds_secrets_and_payload():
    assert MCPErrorCode.AUTHENTICATION.value == "authentication"
    sanitizer = DefaultSecurityEventSanitizer()
    event = {
        "reason": "https://example.test/callback?token=secret-value",
        "nested": ["x" * 4_096 for _ in range(10)],
    }
    result = sanitizer.sanitize(event)
    rendered = str(result)
    assert "secret-value" not in rendered
    assert len(rendered.encode("utf-8")) <= sanitizer._MAX_PAYLOAD


@pytest.mark.asyncio
async def test_mcp_discovery_uses_list_tools_not_contextual_get_tools():
    manager = MCPConnectionManager()
    spec = MCPServerSpec(
        id="srv",
        name="srv",
        transport="stdio",
        command=("x",),
    )

    class Tool:
        name = "query"
        description = "query"
        inputSchema = {"type": "object", "properties": {}}

    class Toolset:
        async def list_tools(self):
            return (Tool(),)

        async def get_tools(self, _ctx):
            raise AssertionError("contextual get_tools must not be used for discovery")

    manager.get_toolset = lambda _spec: _async_handle(Toolset())
    result = await manager.list_tools_result(spec)
    assert result.verified is True
    assert [tool.name for tool in result.tools] == ["query"]


async def _async_handle(toolset):
    return MCPToolsetHandle(MCPConnectionRef("srv", "fp"), toolset)
