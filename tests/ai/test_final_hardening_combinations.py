"""Combination tests for the hardened runtime paths: each test drives a real
multi-component chain end to end (not a helper in isolation)."""

from types import SimpleNamespace

import pytest

from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
from linktools.ai.capability.provider import CapabilityContext
from linktools.ai.capability.models import CapabilityRef
from linktools.ai.events.payloads import TruncatedSecurityEvent
from linktools.ai.mcp.client import MCPConnectionRef
from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPProvider, MCPToolInfo
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.mcp.spec import MCPServerSpecProvider
from linktools.ai.runtime import RuntimeDependencies
from linktools.ai.mcp.codec import parse_mcp_spec
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.governance.security.emitter import (
    DefaultSecurityEventSanitizer,
    EventStoreSecurityEventEmitter,
)
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.tool.executor import GovernedToolInvoker
from linktools.ai.tool.managed import ManagedToolAdapter
from linktools.ai.tool.policy import IdempotencyStrategy, ResolvedToolPolicy


def _descriptor() -> ToolDescriptor:
    return ToolDescriptor(
        name="t", source="test", category="c", risk="low", mutating=False
    )


class _TinySanitizer(DefaultSecurityEventSanitizer):
    # Force the truncation path even for the small audit events the adapter
    # emits, so the full adapter->emitter->sanitizer->store chain is exercised.
    _MAX_PAYLOAD = 64


# --- Combination 1: oversized security event through the full managed chain ---


@pytest.mark.asyncio
async def test_large_security_event_is_persisted_through_managed_adapter(tmp_path):
    store = FilesystemEventStore(root=tmp_path)
    ctx = SimpleNamespace(run_id="r1")
    emitter = EventStoreSecurityEventEmitter(
        store, context=ctx, sanitizer=_TinySanitizer(), failure_mode="fail_closed"
    )

    async def handler(x: str = "d") -> str:
        return f"ok:{x}"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        security_event_emitter=emitter,
        run_context=ctx,
    )

    result = await adapter.invoke(x="hi")  # must not raise TypeError
    assert result == "ok:hi"

    page = await store.list("r1")
    assert page.items, "audit events should have been persisted"
    # Every persisted payload is a valid dataclass -- truncation kept the store
    # contract intact (no dict, no TypeError), in fail_closed mode.
    assert all(isinstance(env.payload, TruncatedSecurityEvent) for env in page.items)


# --- Combination 2: Runtime.inspect + MCP best-effort degradation full chain ---


class _InfoSpecProvider(MCPServerSpecProvider):
    def __init__(self, spec):
        self._spec = spec

    async def list_ids(self):
        return ("risk",)

    async def get(self, server_id):
        return self._spec


class _UnenumerableManager:
    async def list_tools(self, spec):
        return ()

    async def get_toolset(self, spec):
        from pydantic_ai.toolsets import FunctionToolset

        return FunctionToolset()

    async def list_tools_result(self, spec):
        return MCPDiscoveryResult(
            tools=(), verified=False, error=RuntimeError("enumeration unavailable")
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        raise RuntimeError("no tools should be resolved")


@pytest.mark.asyncio
async def test_runtime_inspect_reports_mcp_best_effort_degradation(tmp_path):
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef

    spec_mcp = parse_mcp_spec(
        "risk",
        {
            "transport": "stdio",
            "command": ["python", "-m", "r"],
            "discovery_mode": "best_effort",
        },
    )
    provider = MCPProvider(_InfoSpecProvider(spec_mcp), _UnenumerableManager())

    storage = FilesystemStorage(root=tmp_path)
    rt = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(),
        providers=RuntimeDependencies(capabilities=(provider,)),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )

    agent = AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="mcp", name="risk"),),
    )
    inspection = await rt.inspect(agent)

    assert inspection.tools == ()
    assert any("security degraded" in w for w in inspection.warnings)


# --- Combination 3: dynamic provider illegal BUSINESS_KEY fails before execution ---


@pytest.mark.asyncio
async def test_dynamic_business_key_policy_fails_before_tool_execution():
    class _BusinessKeyNoIdempotentProvider:
        async def resolve(self, descriptor, ctx):
            # Valid tri-state declaration (business_key + field, idempotent left
            # for another layer) that finalizes to an invalid effective policy.
            return ResolvedToolPolicy(
                idempotency_strategy=IdempotencyStrategy.BUSINESS_KEY,
                idempotency_key_field="ext_id",
            )

    handler_calls: "list" = []

    async def handler(**arguments):
        handler_calls.append(arguments)
        return "ok"

    adapter = ManagedToolAdapter(
        descriptor=_descriptor(),
        handler=handler,
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        policy_provider=_BusinessKeyNoIdempotentProvider(),
        run_context=SimpleNamespace(run_id="r1"),
    )

    with pytest.raises(ValueError, match="idempotent=true"):
        await adapter.invoke(x="hi")
    # The handler never ran and no idempotency side effect could occur.
    assert handler_calls == []


# --- Combination 4: MCP tool executes through the managed path to call_tool ---


class _RecordingManager:
    def __init__(self):
        self.calls: "list" = []

    async def list_tools(self, spec):
        return ("get_x",)

    async def get_toolset(self, spec):
        from pydantic_ai.toolsets import FunctionToolset

        return FunctionToolset()

    async def list_tools_result(self, spec):
        info = MCPToolInfo(
            name="get_x", parameters_json_schema={"type": "object", "properties": {}}
        )
        return MCPDiscoveryResult(
            tools=(info,), verified=True, connection_ref=MCPConnectionRef("risk", "fp")
        )

    async def call_tool(self, *, connection_ref, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return {"value": 42}


@pytest.mark.asyncio
async def test_mcp_tool_runs_through_managed_adapter_and_executor_to_call_tool(
    tmp_path,
):
    spec_mcp = parse_mcp_spec(
        "risk", {"transport": "stdio", "command": ["python", "-m", "r"]}
    )
    manager = _RecordingManager()
    provider = MCPProvider(_InfoSpecProvider(spec_mcp), manager)

    bundle = await provider.resolve(
        CapabilityRef("mcp", "risk"),
        CapabilityContext(
            agent_id="a1", exposure_policy=CapabilityToolExposurePolicy()
        ),
    )
    definition = bundle.tool_contributions[0].tools[0]

    executor = GovernedToolInvoker(policy=PolicyEngine(rules=()))
    adapter = ManagedToolAdapter(
        descriptor=definition.descriptor,
        handler=definition.handler,
        tool_executor=executor,
        run_context=SimpleNamespace(run_id="r1"),
    )

    result = await adapter.invoke()

    assert result == {"value": 42}
    # The managed path forwarded to the MCP raw name via call_tool.
    assert manager.calls == [("get_x", {})]
