from types import SimpleNamespace

import acp.schema as schema

from linktools.ai.acp.protocol import AcpMode, CapabilityBuilder, CapabilityInput


def test_capabilities_are_conservative_and_schema_native() -> None:
    values = CapabilityInput(modes=(AcpMode("default", "Default"),))
    capabilities = CapabilityBuilder().build(values, client_capabilities=None)

    assert isinstance(capabilities, schema.AgentCapabilities)
    assert capabilities.prompt_capabilities.image is False
    assert capabilities.mcp_capabilities.http is True
    assert capabilities.mcp_capabilities.sse is True
    assert capabilities.mcp_capabilities.acp is False
    assert capabilities.session_capabilities.additional_directories is None


def test_additional_directories_require_client_filesystem_capability() -> None:
    client = SimpleNamespace(fs=SimpleNamespace(read_text_file=True))
    capabilities = CapabilityBuilder().build(
        CapabilityInput(modes=(AcpMode("default", "Default"),)),
        client_capabilities=client,
    )

    assert capabilities.session_capabilities.additional_directories is not None
