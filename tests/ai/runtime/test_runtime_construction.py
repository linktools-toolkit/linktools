import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.agent.assembly import AgentFeatureRef
from linktools.ai.agent.integrations.mcp import (
    MCPConnectionRef,
    MCPDiscoveryResult,
    MCPRuntimePolicy,
    MCPServerSpec,
    MCPToolInfo,
    MCPToolProvider,
)
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.agent.tool.policy.resolver import ResolvedToolPolicy
from linktools.ai.agent.tool.state.persistence.memory import LocalToolStateBackend
from linktools.ai.agent.tool.state.store import ToolStateStore
from linktools.ai.errors import PrincipalAccessDeniedError
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.runtime import (
    LocalDirectoryStorage,
    RuntimeDependencies,
    RuntimeRequirements,
    build_runtime,
)
from linktools.ai.governance.identity import trusted_local_principal
from tests.ai.fakes.model import make_router


def spec() -> AgentSpec:
    return AgentSpec("agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer"))


@pytest.mark.asyncio
async def test_build_runtime_uses_execution_store_for_local_storage(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    principal = trusted_local_principal(tenant_id="t")
    assert (await runtime.run(spec(), "hello", session_id="s", principal=principal)) is not None
    assert (await runtime.run(spec(), "again", session_id="s", principal=principal)) is not None
    session = await storage.execution.get_session("s")
    assert session.latest_completed_run_id is not None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_wires_session_window_and_emits_event(tmp_path):
    class Window:
        calls = 0

        async def select_messages(self, messages, model_policy):
            self.calls += 1
            return messages

    class Events:
        events = []

        async def emit(self, event):
            self.events.append(event)

    window = Window()
    events = Events()
    runtime = build_runtime(
        storage=LocalDirectoryStorage(tmp_path),
        dependencies=RuntimeDependencies(
            model_resolver=make_router(),
            session_window=window,
            security_events=events,
        ),
    )
    await runtime.run(
        spec(),
        "hello",
        principal=trusted_local_principal(tenant_id="t"),
    )
    assert window.calls == 1
    assert any(
        type(event).__name__ == "PromptWindowApplied"
        for event in events.events
    )
    await runtime.aclose()


@pytest.mark.asyncio
async def test_anonymous_runs_get_isolated_sessions(tmp_path):
    # Omitting session_id must NOT collapse every run onto a shared "session".
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    principal = trusted_local_principal()
    assert (await runtime.run(spec(), "hello", principal=principal)) is not None
    assert (await runtime.run(spec(), "again", principal=principal)) is not None
    assert await storage.execution.get_session("session") is None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_session_reuse_rejects_principal_mismatch(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    await runtime.run(spec(), "hi", session_id="s", principal=trusted_local_principal(tenant_id="t1"))
    with pytest.raises((PrincipalAccessDeniedError, Exception)) as caught:
        await runtime.run(spec(), "hi", session_id="s", principal=trusted_local_principal(tenant_id="t2"))
    assert type(caught.value).__name__ in {
        "PrincipalAccessDeniedError",
        "StorageConflictError",
    }
    await runtime.aclose()


@pytest.mark.asyncio
async def test_runtime_executes_mcp_tool_through_governed_main_chain(
    tmp_path,
):
    class Specs:
        async def list_ids(self):
            return ("server",)

        async def get(self, server_id):
            assert server_id == "server"
            return MCPServerSpec(
                "server",
                "server",
                "stdio",
                command=("fake",),
                tool_prefix=False,
            )

    class Connections:
        calls = []
        closed = False

        async def list_tools_result(self, server):
            return MCPDiscoveryResult(
                tools=(
                    MCPToolInfo(
                        "echo",
                        {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        read_only=True,
                    ),
                ),
                verified=True,
                connection_ref=MCPConnectionRef(server.id, "revision"),
            )

        async def call_tool(
            self, *, connection_ref, tool_name, arguments
        ):
            self.calls.append((connection_ref, tool_name, arguments))
            return {"echo": arguments["text"]}

        async def close(self):
            self.closed = True

    class Policy:
        async def resolve(self, descriptor, context):
            return ResolvedToolPolicy(enabled=True)

    model_calls = 0

    def model(messages, info):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="echo",
                        args={"text": "hello"},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                TextPart(
                    content='{"response":{"message":"mcp complete"}}'
                )
            ]
        )

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model))
    connections = Connections()
    state = LocalToolStateBackend()
    storage = LocalDirectoryStorage(tmp_path, tools=state)
    runtime = build_runtime(
        storage=storage,
        dependencies=RuntimeDependencies(
            model_resolver=ModelResolver(registry=registry),
            tool_policy=Policy(),
            mcp_provider=MCPToolProvider(Specs(), connections),
            mcp_policy=MCPRuntimePolicy(),
        ),
        requirements=RuntimeRequirements(tools=True),
    )
    agent = AgentSpec(
        "agent",
        "agent",
        ModelPolicy(primary="test-model"),
        PromptSpec("use echo"),
        features=(AgentFeatureRef("mcp", "server"),),
    )
    principal = trusted_local_principal(tenant_id="tenant")

    result = await runtime.run(
        agent,
        "echo hello",
        session_id="session",
        execution_id="run-mcp",
        principal=principal,
    )

    assert result is not None
    assert connections.calls
    detail = await runtime.inspect(run_id="run-mcp", principal=principal)
    assert len(detail.tool_calls) == 1
    assert detail.tool_calls[0].status == "completed"
    await runtime.aclose()
    assert connections.closed
