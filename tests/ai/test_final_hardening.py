import pytest

from linktools.ai.errors import (
    IdempotencyConfigurationError,
)
from linktools.ai.mcp.provider import MCPExposedTool
from linktools.ai.security.descriptor import ToolDescriptor
from linktools.ai.security.redact import redact_for_audit, redact_exception
from linktools.ai.tool.idempotency_key import DefaultIdempotencyKeyBuilder
from linktools.ai.tool.policy import EffectiveToolPolicy, IdempotencyStrategy


def test_public_snapshots_freeze_nested_values_and_copy_inputs():
    metadata = {"nested": {"items": [1]}}
    descriptor = ToolDescriptor(
        name="tool", source="test", category="discovery", risk="low",
        mutating=False, metadata=metadata,
    )
    metadata["nested"]["items"].append(2)
    assert tuple(descriptor.metadata["nested"]["items"]) == (1,)
    with pytest.raises(TypeError):
        descriptor.metadata["nested"]["items"] = ()


def test_business_key_requires_trusted_configured_field():
    descriptor = ToolDescriptor(
        name="create", source="test", category="mcp-write", risk="high",
        mutating=True,
    )
    policy = EffectiveToolPolicy(
        idempotent=True,
        idempotency_strategy=IdempotencyStrategy.BUSINESS_KEY,
        idempotency_key_field="external_id",
    )
    with pytest.raises(IdempotencyConfigurationError):
        DefaultIdempotencyKeyBuilder().build(
            descriptor=descriptor, arguments={}, run_context=type("C", (), {"run_id": "r"})(),
            schema_version="1", policy=policy,
        )


def test_redaction_recurses_and_masks_exception_tokens():
    value = redact_for_audit({"nested": [{"authorization": "Bearer abc"}]})
    assert value["nested"][0]["authorization"] == "***REDACTED***"
    assert "Bearer abc" not in redact_exception(RuntimeError("Bearer abc"))


def test_mcp_exposed_tool_is_deeply_immutable():
    tool = MCPExposedTool("server", "raw", "server.raw", metadata={"x": {"y": 1}})
    with pytest.raises(TypeError):
        tool.metadata["x"]["y"] = 2
