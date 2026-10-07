#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real localhost MCP protocol, execution boundary, and recovery contracts."""

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from mcp import types as mcp_types

try:
    from mcp.server.mcpserver import MCPServer, Context
except ImportError:
    from mcp.server.fastmcp import Context, FastMCP as MCPServer

    _MODERN_SERVER = False
else:
    _MODERN_SERVER = True
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, JsonValue, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage, ToolEffectApplied
from linktools.ai.runtime._mcp import (
    _MCPBinding,
    _raise_primary_after_cleanup,
    close_mcp_resources,
    materialize_mcp_capabilities,
)
from linktools.ai.spec import MCPServerSpec, mcp_server_selector, mcp_tool_selector
from linktools.ai.workspace import (
    BubblewrapSandbox,
    LocalSandbox,
    SandboxResource,
    SandboxSession,
)

from ._runtime_test_helpers import RuntimeUsageModels, _UsageFunctionModel
from .test_tool_effect_semantics import _Bridge


@pytest.fixture(autouse=True)
def _local_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path / "linktools"))
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


class _RemoteServer:
    """Serve the SDK's HTTP protocols and inject failures at the HTTP boundary."""

    def __init__(self, transport: str, *, json_response: bool = True) -> None:
        self.transport = transport
        self.methods: list[str] = []
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.response_types: list[str] = []
        self.effects: list[str] = []
        self.call_headers: list[tuple[str, dict[str, str]]] = []
        self.active_requests = 0
        self.active_streams = 0
        self.auth: str | None = None
        self.fail_method: str | None = None
        self.malformed_method: str | None = None
        self.duplicate_tools = False
        self.delay_method: str | None = None
        self.lose_result = False
        self.block_calls = False
        self.call_started = asyncio.Event()
        self.release_call = asyncio.Event()
        self.url = ""
        self.server = (
            MCPServer("linktools-http-test")
            if _MODERN_SERVER
            else MCPServer("linktools-http-test", json_response=json_response)
        )

        @self.server.tool()
        async def echo(ctx: Context, value: str = "hello") -> str:
            self.effects.append(value)
            if not json_response:
                await ctx.report_progress(0, 1, "tool response is streaming")
            self.call_started.set()
            if self.block_calls:
                await self.release_call.wait()
            return f"echo:{value}"

        @self.server.tool()
        async def hidden() -> str:
            return "not selected"

        @self.server.tool()
        async def refresh(ctx: Context) -> str:
            @self.server.tool()
            async def added() -> str:
                return "new tool"

            await ctx.session.send_tool_list_changed()
            return "catalog changed"

        self.app: ASGIApp = (
            self.server.sse_app()
            if transport == "sse"
            else (
                self.server.streamable_http_app(json_response=json_response)
                if _MODERN_SERVER
                else self.server.streamable_http_app()
            )
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        self.active_requests += 1
        streaming = scope["method"] == "GET"
        self.active_streams += int(streaming)
        headers = {
            key.decode().lower(): value.decode() for key, value in scope["headers"]
        }
        self.requests.append((scope["method"], headers))
        try:
            if self.auth is not None and headers.get("authorization") != self.auth:
                await PlainTextResponse("private-auth-response", status_code=401)(
                    scope,
                    receive,
                    send,
                )
                return
            method = ""
            if scope["method"] == "POST":
                body = bytearray()
                while True:
                    message = await receive()
                    body.extend(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
                payload = json.loads(body)
                method = payload.get("method", "")
                self.methods.append(method)
                if method == "tools/call":
                    self.call_headers.append(
                        (payload["params"]["arguments"].get("value", ""), headers)
                    )
                original_receive = receive
                delivered = False

                async def replay_receive() -> Message:
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body)}
                    return await original_receive()

                receive = replay_receive
                if self.duplicate_tools and method == "tools/list":
                    tool = {
                        "name": "duplicate",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                    await JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": mcp_types.ListToolsResult(
                                tools=[mcp_types.Tool(**tool), mcp_types.Tool(**tool)],
                            ).model_dump(by_alias=True, exclude_none=True),
                        }
                    )(scope, receive, send)
                    return
                if method == self.malformed_method or (
                    self.malformed_method == "negotiation"
                    and method in {"server/discover", "initialize"}
                ):
                    await JSONResponse(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {
                                "tools": "private-malformed-response",
                                "capabilities": "private-malformed-response",
                            },
                        }
                    )(scope, receive, send)
                    return
                if method == self.fail_method:
                    await PlainTextResponse(
                        "private-protocol-response", status_code=503
                    )(
                        scope,
                        receive,
                        send,
                    )
                    return
                if method == self.delay_method:
                    await self.release_call.wait()

            async def record_send(message: Message) -> None:
                if message["type"] == "http.response.start":
                    response_headers = dict(message.get("headers", ()))
                    self.response_types.append(
                        response_headers.get(b"content-type", b"").decode()
                    )
                if self.lose_result and method == "tools/call":
                    if message["type"] == "http.response.body" and not message.get(
                        "more_body", False
                    ):
                        await PlainTextResponse("private-lost-result", status_code=503)(
                            scope,
                            receive,
                            send,
                        )
                    return
                await send(message)

            await self.app(scope, receive, record_send)
        finally:
            self.active_requests -= 1
            self.active_streams -= int(streaming)


@asynccontextmanager
async def _serve(
    transport: str = "streamable-http",
    *,
    json_response: bool = True,
) -> AsyncIterator[_RemoteServer]:
    remote = _RemoteServer(transport, json_response=json_response)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    port = sock.getsockname()[1]
    remote.url = f"http://127.0.0.1:{port}/{'sse' if transport == 'sse' else 'mcp'}"
    server = uvicorn.Server(
        uvicorn.Config(
            remote,
            lifespan="on",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=1,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:

        async def wait_started() -> None:
            while not server.started:
                if task.done():
                    await task
                    raise AssertionError("localhost MCP server did not start")
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_started(), timeout=5)
        yield remote
    finally:
        remote.release_call.set()
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            sock.close()


async def _wait_closed(remote: _RemoteServer) -> None:
    async def wait_closed() -> None:
        while remote.active_requests:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait_closed(), timeout=5)


def _context() -> RunContext[object]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="mcp-run",
        tool_call_id="mcp-call",
    )


@asynccontextmanager
async def _toolsets(
    *servers: MCPServerSpec,
    selectors: tuple[str, ...] | None = None,
) -> AsyncIterator[tuple[Any, ...]]:
    selected = selectors or tuple(mcp_server_selector(server.id) for server in servers)
    capabilities = await materialize_mcp_capabilities(
        servers,
        selected,
        sandbox=None,
        sandbox_session=None,
        host_cwd=None,
        bindings={
            server.id: _MCPBinding(
                None, None, {"version": 1, "boundary": "host-network"}
            )
            for server in servers
        },
        projections={},
        tool_operations=_Bridge(False),
        tool_metrics=None,
    )
    primary: BaseException | None = None
    try:
        async with AsyncExitStack() as stack:
            toolsets = tuple(capability.get_toolset() for capability in capabilities)
            for toolset in toolsets:
                await stack.enter_async_context(toolset)
            yield toolsets
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            await close_mcp_resources(capabilities)
        except BaseException as cleanup:
            if primary is not None:
                _raise_primary_after_cleanup(primary, cleanup)
            raise


def _spec(
    remote: _RemoteServer, server_id: str = "remote", **kwargs: Any
) -> MCPServerSpec:
    return MCPServerSpec(
        server_id,
        transport=remote.transport,
        url=remote.url,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "json_response", "response_type"),
    (
        ("streamable-http", True, "application/json"),
        ("streamable-http", False, "text/event-stream"),
        ("sse", False, "text/event-stream"),
    ),
)
async def test_remote_protocol_discovers_calls_authenticates_and_closes(
    transport: str,
    json_response: bool,
    response_type: str,
) -> None:
    async with _serve(transport, json_response=json_response) as remote:
        remote.auth = "Bearer test-service-token"
        async with _toolsets(
            _spec(
                remote, headers={"Authorization": remote.auth, "X-Caller": "linktools"}
            ),
            selectors=(mcp_tool_selector("remote", "echo"),),
        ) as (toolset,):
            tools = await toolset.get_tools(_context())
            assert len(tools) == 1
            name, tool = next(iter(tools.items()))
            assert name.startswith("mcp__")
            assert '"tool_name":"echo"' in tool.tool_def.description
            assert '"tool_name":"hidden"' not in tool.tool_def.description
            result = await toolset.call_tool(name, {"value": "hello"}, _context(), tool)
            assert "echo:hello" in str(result)
        await _wait_closed(remote)
        assert "tools/list" in remote.methods
        assert "tools/call" in remote.methods
        assert remote.effects == ["hello"]
        assert any(response_type in value for value in remote.response_types)
        assert all(
            headers.get("authorization") == remote.auth
            for _, headers in remote.requests
        )
        assert all(
            headers.get("x-caller") == "linktools" for _, headers in remote.requests
        )
        if transport == "streamable-http" and any(
            "mcp-session-id" in headers for _, headers in remote.requests
        ):
            assert any(method == "DELETE" for method, _ in remote.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("authentication", "initialize", "tools/list"))
async def test_connection_failures_are_typed_and_redact_remote_secrets(
    failure: str,
) -> None:
    async with _serve(
        "sse" if failure == "initialize" else "streamable-http"
    ) as remote:
        remote.url += "?private-query-token"
        if failure == "authentication":
            remote.auth = "Bearer required-private-token"
        else:
            remote.fail_method = failure
        with pytest.raises(AIError) as raised:
            async with _toolsets(
                _spec(
                    remote,
                    headers={"Authorization": "Bearer wrong-private-token"},
                    init_timeout=0.3,
                    read_timeout=0.2,
                )
            ) as (toolset,):
                await toolset.get_tools(_context())
        error = raised.value
        assert error.code is ErrorCode.MCP_CONNECTION_FAILED
        assert error.safe_details["server_id"] == "remote"
        assert error.safe_details["transport"] == remote.transport
        assert "phase" in error.safe_details
        diagnostic = json.dumps(error.safe_details) + str(error)
        for secret in (
            remote.url,
            "private-query-token",
            "wrong-private-token",
            "private-protocol-response",
            "private-auth-response",
        ):
            assert secret not in diagnostic
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_initialization_timeout_closes_remote_connection() -> None:
    async with _serve("sse") as remote:
        remote.delay_method = "initialize"
        with pytest.raises(AIError) as raised:
            async with _toolsets(_spec(remote, init_timeout=0.05, read_timeout=0.1)):
                pytest.fail("initialization must time out")
        assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
        remote.release_call.set()
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_partial_remote_initialization_closes_already_opened_connections() -> (
    None
):
    async with _serve() as first, _serve() as second:
        second.auth = "Bearer required-token"
        with pytest.raises(AIError) as raised:
            async with _toolsets(_spec(first, "first"), _spec(second, "second")):
                pytest.fail("second connection must fail")
        assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
        await _wait_closed(first)
        await _wait_closed(second)
        assert first.active_streams == 0
        assert first.effects == second.effects == []


@pytest.mark.asyncio
async def test_explicit_missing_remote_tool_fails_instead_of_disappearing() -> None:
    async with _serve() as remote:
        with pytest.raises(AIError) as raised:
            async with _toolsets(
                _spec(remote),
                selectors=(mcp_tool_selector("remote", "missing"),),
            ) as (toolset,):
                await toolset.get_tools(_context())
        assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
        assert remote.effects == []
        await _wait_closed(remote)


class _RemoteModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:remote-mcp"
    vision = False
    contract: dict[str, JsonValue] = {"provider": "test", "model": "remote-mcp"}

    def materialize(self) -> _UsageFunctionModel:
        async def request(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            returned = {
                part.tool_call_id
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, ToolReturnPart) and part.outcome != "interrupted"
            }
            calls = [
                part
                for message in messages
                if isinstance(message, ModelResponse)
                for part in message.parts
                if isinstance(part, ToolCallPart)
            ]
            pending = next(
                (call for call in reversed(calls) if call.tool_call_id not in returned),
                None,
            )
            if not calls or pending is not None:
                name = (
                    pending.tool_name
                    if pending
                    else next(
                        tool.name
                        for tool in info.function_tools
                        if '"tool_name":"echo"' in (tool.description or "")
                    )
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            name,
                            {"value": "committed"},
                            tool_call_id=pending.tool_call_id
                            if pending
                            else "remote-effect",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("done")])

        return _UsageFunctionModel(request)


class _RemoteModels(RuntimeUsageModels):
    def resolve(self, route_id: str) -> _RemoteModelBinding:
        assert route_id == "default"
        return _RemoteModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _RemoteModelBinding:
        assert route_id in (None, "default")
        assert dict(payload) == _RemoteModelBinding.contract
        return _RemoteModelBinding()


def _group(*servers: MCPServerSpec, sandbox: Any = None) -> CapabilityGroup[object]:
    group = CapabilityGroup[object]("remote", sandbox=sandbox)
    for server in servers:
        group.mcp(server)
    group.agent("default", allow_tools=(mcp_tool_selector(servers[0].id, "echo"),))
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_kind", ("local", "bubblewrap"))
async def test_remote_only_runtime_uses_host_network_without_opening_sandbox(
    sandbox_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = (
        LocalSandbox()
        if sandbox_kind == "local"
        else BubblewrapSandbox(
            runtime_root=tmp_path, bwrap_executable=tmp_path / "bwrap"
        )
    )

    async def forbidden_open(
        self: object,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del self, root, resources
        pytest.fail("remote-only execution must not open a sandbox session")

    monkeypatch.setattr(type(sandbox), "open", forbidden_open)
    async with _serve() as remote:
        async with Runtime.open(
            "remote-only",
            models=_RemoteModels(),
            storage=RuntimeStorage.in_memory(),
            capabilities=(_group(_spec(remote), sandbox=sandbox),),
        ) as runtime:
            result = (await runtime.agents.get("default").run("echo", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
        assert remote.effects == ["committed"]
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_unselected_offline_remote_does_not_connect() -> None:
    async with _serve() as remote:
        offline = MCPServerSpec(
            "offline", transport="streamable-http", url="http://127.0.0.1:1/mcp"
        )
        async with Runtime.open(
            "remote-selection",
            models=_RemoteModels(),
            storage=RuntimeStorage.in_memory(),
            capabilities=(_group(_spec(remote), offline),),
        ) as runtime:
            result = (await runtime.agents.get("default").run("echo", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
        assert remote.effects == ["committed"]


@pytest.mark.asyncio
async def test_lost_remote_result_requires_confirmation_and_recovery_does_not_repeat_effect(
    tmp_path: Path,
) -> None:
    async with _serve() as remote:
        remote.lose_result = True
        state_path = tmp_path / "runtime"
        storage = RuntimeStorage.filesystem(state_path)
        async with Runtime.open(
            "remote-recovery",
            models=_RemoteModels(),
            storage=storage,
            capabilities=(_group(_spec(remote)),),
        ) as runtime:
            execution = await runtime.agents.get("default").start("echo")
            with pytest.raises(AIError) as raised:
                await execution.wait(timeout_seconds=10)
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            with pytest.raises(AIError) as unresolved:
                await execution.recover()
            assert unresolved.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            effects = await execution.recovery_effects()
            assert len(effects) == 1
            effect = effects[0]
            execution_id = execution.execution_id
            assert remote.effects == ["committed"]
        await storage.close()
        remote.lose_result = False
        restored = RuntimeStorage.filesystem(state_path)
        try:
            async with Runtime.open(
                "remote-recovery",
                models=_RemoteModels(),
                storage=restored,
                capabilities=(_group(_spec(remote, headers={"X-Rotated": "true"})),),
            ) as runtime:
                execution = await runtime.executions.get(execution_id)
                resolution = await execution.resolve_tool_effect(
                    effect.operation_id,
                    expected_fence=effect.fence,
                    resolution=ToolEffectApplied("echo:committed"),
                    idempotency_key="confirm-remote",
                )
                assert resolution.status is ToolOperationStatus.COMPLETED
                await execution.recover()
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.SUCCEEDED
                assert remote.effects == ["committed"]
        finally:
            await restored.close()
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_independent_clients_keep_credentials_and_survive_peer_close() -> None:
    async with _serve("sse", json_response=False) as remote:
        remote.block_calls = True
        async with _toolsets(
            _spec(remote, headers={"Authorization": "Bearer second"})
        ) as (second,):
            second_tools = await second.get_tools(_context())
            second_name, second_tool = next(
                (name, tool)
                for name, tool in second_tools.items()
                if '"tool_name":"echo"' in tool.tool_def.description
            )
            pending = asyncio.create_task(
                second.call_tool(
                    second_name,
                    {"value": "second"},
                    _context(),
                    second_tool,
                )
            )
            try:
                await asyncio.wait_for(remote.call_started.wait(), timeout=5)
                async with _toolsets(
                    _spec(remote, headers={"Authorization": "Bearer first"})
                ) as (first,):
                    assert await first.get_tools(_context())
                assert not pending.done()
                remote.release_call.set()
                assert "echo:second" in str(await asyncio.wait_for(pending, timeout=5))
            finally:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        await _wait_closed(remote)
        assert remote.effects == ["second"]
        assert remote.call_headers[0][1]["authorization"] == "Bearer second"
        assert {headers.get("authorization") for _, headers in remote.requests} == {
            "Bearer first",
            "Bearer second",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("streamable-http", "sse"))
async def test_cancelled_remote_leaf_closes_connection_without_repeating_effect(
    transport: str,
) -> None:
    async with _serve(transport) as remote:
        remote.block_calls = True

        async def call() -> None:
            async with _toolsets(
                _spec(remote),
                selectors=(mcp_tool_selector("remote", "echo"),),
            ) as (toolset,):
                tools = await toolset.get_tools(_context())
                name, tool = next(iter(tools.items()))
                await toolset.call_tool(name, {"value": "cancelled"}, _context(), tool)

        task = asyncio.create_task(call())
        await asyncio.wait_for(remote.call_started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        remote.release_call.set()
        await _wait_closed(remote)
        assert remote.effects == ["cancelled"]


@pytest.mark.asyncio
async def test_parallel_runtimes_do_not_share_service_credentials() -> None:
    async with _serve() as remote:

        async def run(label: str) -> None:
            async with Runtime.open(
                f"remote-{label}",
                models=_RemoteModels(),
                storage=RuntimeStorage.in_memory(),
                capabilities=(
                    _group(_spec(remote, headers={"Authorization": f"Bearer {label}"})),
                ),
            ) as runtime:
                result = (await runtime.agents.get("default").run(
                    "echo", timeout_seconds=10
                )).result
                assert result.status is ExecutionStatus.SUCCEEDED

        await asyncio.gather(run("first"), run("second"))
        assert remote.effects == ["committed", "committed"]
        assert {headers["authorization"] for _, headers in remote.call_headers} == {
            "Bearer first",
            "Bearer second",
        }
        await _wait_closed(remote)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("negotiation", "tools/list"))
async def test_malformed_protocol_responses_are_typed_without_disclosing_body(
    phase: str,
) -> None:
    async with _serve() as remote:
        remote.malformed_method = phase
        with pytest.raises(AIError) as raised:
            async with _toolsets(_spec(remote, init_timeout=0.3, read_timeout=0.2)) as (
                toolset,
            ):
                await toolset.get_tools(_context())
        assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
        assert raised.value.safe_details["server_id"] == "remote"
        assert "private-malformed-response" not in str(raised.value)
        assert "private-malformed-response" not in json.dumps(raised.value.safe_details)
        await _wait_closed(remote)


class _RecordingLocalSandbox(LocalSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.roots: list[Path] = []

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        self.roots.append(root)
        return await super().open(root=root, resources=resources)


class _MixedModelBinding(_RemoteModelBinding):
    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="done")


class _MixedModels(_RemoteModels):
    def resolve(self, route_id: str) -> _MixedModelBinding:
        assert route_id == "default"
        return _MixedModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _MixedModelBinding:
        assert route_id in (None, "default")
        assert dict(payload) == _MixedModelBinding.contract
        return _MixedModelBinding()


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_kind", ("local", "bubblewrap"))
async def test_mixed_stdio_and_http_keep_separate_process_and_network_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sandbox_kind: str,
) -> None:
    import sys

    if sandbox_kind == "bubblewrap":
        from .test_bubblewrap_integration import _sandbox_configuration

        runtime_root, executable = _sandbox_configuration()
        sandbox = BubblewrapSandbox(
            runtime_root=runtime_root, bwrap_executable=executable
        )
        command, argument = "/usr/bin/python3", "/workspace/server.py"
    else:
        sandbox = _RecordingLocalSandbox()
        command, argument = sys.executable, str(tmp_path / "server.py")
    script = tmp_path / "server.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "from pathlib import Path\n"
        "import json, os, socket\n"
        "try:\n"
        "    from mcp.server.mcpserver import MCPServer\n"
        "except ImportError:\n"
        "    from mcp.server.fastmcp import FastMCP as MCPServer\n"
        "server = MCPServer('local-test')\n"
        "@server.tool()\n"
        "def record() -> str:\n"
        "    try:\n"
        "        with socket.create_connection(('127.0.0.1', int(os.environ['REMOTE_PORT'])), timeout=1):\n"
        "            network = 'allowed'\n"
        "    except OSError:\n"
        "        network = 'blocked'\n"
        "    Path('local-effect.txt').write_text(json.dumps({'cwd': str(Path.cwd()), 'network': network}))\n"
        "    return 'local-result'\n"
        "server.run(transport='stdio')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    async with _serve() as remote:
        group = CapabilityGroup[object]("mixed", sandbox=sandbox)
        group.mcp(_spec(remote))
        group.mcp(
            MCPServerSpec(
                "local",
                command,
                (argument,),
                env={"REMOTE_PORT": remote.url.split(":")[2].split("/")[0]},
            )
        )
        group.agent(
            "default",
            allow_tools=(
                mcp_tool_selector("remote", "echo"),
                mcp_tool_selector("local", "record"),
            ),
        )
        async with Runtime.open(
            "mixed-mcp",
            models=_MixedModels(),
            storage=RuntimeStorage.in_memory(),
            capabilities=(group,),
        ) as runtime:
            result = (await runtime.agents.get("default").run(
                "call both", timeout_seconds=15
            )).result
            assert result.status is ExecutionStatus.SUCCEEDED
        if isinstance(sandbox, _RecordingLocalSandbox):
            assert sandbox.roots == [tmp_path]
        assert json.loads((tmp_path / "local-effect.txt").read_text()) == {
            "cwd": str(tmp_path) if sandbox_kind == "local" else "/workspace",
            "network": "allowed" if sandbox_kind == "local" else "blocked",
        }
        assert len(remote.effects) == 1
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_remote_duplicate_tool_names_fail_closed() -> None:
    async with _serve() as remote:
        remote.duplicate_tools = True
        with pytest.raises(AIError) as raised:
            async with _toolsets(_spec(remote)) as (toolset,):
                await toolset.get_tools(_context())
        assert raised.value.code is ErrorCode.CAPABILITY_CONFLICT
        assert not remote.effects
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_legacy_remote_catalog_notification_invalidates_sdk_tool_cache() -> None:
    async with _serve("sse") as remote:
        async with _toolsets(_spec(remote)) as (toolset,):
            before = await toolset.get_tools(_context())
            assert not any(
                '"tool_name":"added"' in tool.tool_def.description
                for tool in before.values()
            )
            name, tool = next(
                (name, tool)
                for name, tool in before.items()
                if '"tool_name":"refresh"' in tool.tool_def.description
            )
            await toolset.call_tool(name, {}, _context(), tool)

            async def discover_added() -> None:
                while True:
                    tools = await toolset.get_tools(_context())
                    added = next(
                        (
                            (name, tool)
                            for name, tool in tools.items()
                            if '"tool_name":"added"' in tool.tool_def.description
                        ),
                        None,
                    )
                    if added is not None:
                        added_name, added_tool = added
                        result = await toolset.call_tool(
                            added_name, {}, _context(), added_tool
                        )
                        assert "new tool" in str(result)
                        return
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(discover_added(), timeout=5)
        await _wait_closed(remote)


@pytest.mark.asyncio
async def test_cleanup_failure_preserves_confirmed_tool_result_and_idempotent_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastmcp import Client

    original_close = Client.close
    failed = False

    async def failing_close(client: Client) -> None:
        nonlocal failed
        await original_close(client)
        if not failed:
            failed = True
            raise RuntimeError("private-cleanup-response")

    monkeypatch.setattr(Client, "close", failing_close)
    async with _serve() as remote:
        storage = RuntimeStorage.in_memory()
        async with Runtime.open(
            "remote-cleanup",
            models=_RemoteModels(),
            storage=storage,
            capabilities=(_group(_spec(remote)),),
        ) as runtime:
            execution = await runtime.agents.get("default").start(
                "echo",
                idempotency_key="confirmed-before-close",
            )
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.MCP_CLEANUP_FAILED.value
            assert "private-cleanup-response" not in str(result)
            operations = await storage.recovery.tools.list_by_execution(
                execution.execution_id,
                tenant_id=runtime.default_principal.tenant_id,
            )
            assert len(operations) == 1
            assert operations[0].status is ToolOperationStatus.COMPLETED
            assert operations[0].replay_safe is False
            with pytest.raises(AIError) as replay_error:
                await runtime.agents.get("default").start(
                    "echo",
                    idempotency_key="confirmed-before-close",
                )
            assert replay_error.value.code is ErrorCode.MCP_CLEANUP_FAILED
            assert remote.effects == ["committed"]
        await _wait_closed(remote)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_failure", (False, True))
async def test_one_failed_client_cleanup_still_closes_other_clients(
    monkeypatch: pytest.MonkeyPatch,
    primary_failure: bool,
) -> None:
    from fastmcp import Client

    original_close = Client.close
    closed: list[Client] = []

    async def fail_first_close(client: Client) -> None:
        await original_close(client)
        closed.append(client)
        if len(closed) == 1:
            raise RuntimeError("private-cleanup-response")

    monkeypatch.setattr(Client, "close", fail_first_close)
    async with _serve("sse") as first, _serve("sse") as second:
        with pytest.raises(AIError) as raised:
            async with _toolsets(
                _spec(first, "first"), _spec(second, "second")
            ) as toolsets:
                for toolset in toolsets:
                    assert await toolset.get_tools(_context())
                if primary_failure:
                    raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        assert raised.value.code is (
            ErrorCode.OUTPUT_CONTRACT_INVALID
            if primary_failure
            else ErrorCode.MCP_CLEANUP_FAILED
        )
        assert len({id(client) for client in closed}) == 2
        assert "private-cleanup-response" not in str(raised.value)
        await _wait_closed(first)
        await _wait_closed(second)


@pytest.mark.asyncio
async def test_mixed_runtime_does_not_bypass_failed_local_sandbox() -> None:
    class UnavailableSandbox(LocalSandbox):
        async def open(
            self,
            *,
            root: Path,
            resources: tuple[SandboxResource, ...] = (),
        ) -> SandboxSession:
            del root, resources
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)

    async with _serve() as remote:
        group = CapabilityGroup[object](
            "mixed-unavailable", sandbox=UnavailableSandbox()
        )
        group.mcp(_spec(remote))
        group.mcp(MCPServerSpec("local", "unavailable-local-mcp"))
        group.agent(
            "default",
            allow_tools=(
                mcp_tool_selector("remote", "echo"),
                mcp_server_selector("local"),
            ),
        )
        async with Runtime.open(
            "mixed-unavailable",
            models=_MixedModels(),
            storage=RuntimeStorage.in_memory(),
            capabilities=(group,),
        ) as runtime:
            result = (await runtime.agents.get("default").run(
                "call both", timeout_seconds=10
            )).result
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.SANDBOX_UNAVAILABLE.value
        assert remote.effects == []
        await _wait_closed(remote)
