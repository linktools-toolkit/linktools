#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One Runtime result boundary for ordinary, workspace, and MCP tools."""

import asyncio
import base64
import hashlib
import json
import sqlite3
import sys
import textwrap
import zlib
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import BinaryContent, ToolReturn, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.agent import AgentCompiler
from linktools.ai.asset import AssetKey, AssetStore, AssetVersionRef, InMemoryAssetBackend
from linktools.ai.capability import (
    AgentContext, AssetCapability, CapabilityContribution, CapabilityLoadContext,
    CheckpointCapability, ExecutionCapability, ReferencedInputsCapability,
    ToolCallRetry, TraceCapability, ToolResultsCapability, WorkspaceCapability,
    TOOL_MCP_SELECTOR_METADATA_KEY,
)
from linktools.ai.core import ExecutionStatus, ToolOperationStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage
from linktools.ai.runtime._run_callbacks import RunCallbacks
from linktools.ai.spec import AgentSpec, CapabilityRef, MCPServerSpec
from linktools.ai.storage import StorageEntryRevision, StorageOverlay
from linktools.ai.workspace import Workspace

_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6SgAAAABJRU5ErkJggg=="
_PNG = base64.b64decode(_PNG_BASE64)
_TEXT = "Recoverable synthetic document body, not a durable execution receipt."
_SOURCE = CapabilityRef("capability", "rendered-inputs", 1)
_REFERENCE = ReferencedContent(
    AssetVersionRef(AssetKey("file", "document/page-1"), "renderer",
                    StorageEntryRevision(1), hashlib.sha256(_PNG).hexdigest(), len(_PNG)),
    media_type="image/png", source=_SOURCE,
)


def _server(tmp_path: Path, *, text: bool = False, **kwargs: object) -> MCPServerSpec:
    content = ({"type": "text", "text": _TEXT} if text else
               {"type": "image", "mimeType": "image/png", "data": _PNG_BASE64})
    script = tmp_path / "server.py"
    script.write_text(textwrap.dedent(f'''\
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-
        import json
        import sys
        from pathlib import Path
        for line in sys.stdin:
            request = json.loads(line)
            if "id" not in request:
                continue
            method = request["method"]
            if method == "initialize":
                result = {{"protocolVersion": request["params"]["protocolVersion"],
                          "capabilities": {{"tools": {{}}}},
                          "serverInfo": {{"name": "synthetic", "version": "1"}}}}
            elif method == "tools/list":
                result = {{"tools": [{{"name": "render", "description": "Render one synthetic page.",
                    "_meta": {{"linktools.ai.mcp_selector": "mcp:forged:target"}},
                    "inputSchema": {{"type": "object", "properties": {{"page": {{"type": "integer"}}}},
                                    "required": ["page"], "additionalProperties": False}}}}]}}
            elif method == "tools/call":
                assert request["params"]["name"] == "render"
                assert request["params"]["arguments"] == {{"page": 1}}
                with Path({str(tmp_path / 'calls')!r}).open("a") as stream:
                    stream.write("render\\n")
                result = {{"content": [{content!r}], "isError": False}}
            elif method == "ping":
                result = {{}}
            else:
                print(json.dumps({{"jsonrpc": "2.0", "id": request["id"],
                                  "error": {{"code": -32601, "message": "Method not found"}}}}), flush=True)
                continue
            print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}), flush=True)
    '''), encoding="utf-8")
    return MCPServerSpec("renderer", sys.executable, (str(script),), **kwargs)


class _Binding:
    provider = "test"
    model_identity = "test:result-projection"
    contract = {"provider": provider, "model_identity": model_identity,
                "input_mime_types": ["image/png"]}

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def materialize(self) -> FunctionModel:
        return FunctionModel(stream_function=self.stream)


def _stored_bytes(root: Path) -> bytes:
    """Inspect physical objects and decoded SQLite inline payloads."""
    blobs = [path.read_bytes() for path in root.rglob("*") if path.is_file()]

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if value.get("kind") == "inline" and value.get("encoding") == "base64":
                raw = base64.b64decode(value["value"])
                blobs.append(raw)
                try:
                    walk(json.loads(raw))
                except (UnicodeError, ValueError):
                    pass
            payload = value.get("payload")
            if value.get("compression") == "zlib" and isinstance(payload, dict):
                raw = zlib.decompress(base64.b64decode(payload["value"]))
                blobs.append(raw)
                walk(json.loads(raw))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for raw in tuple(blobs):
        try:
            walk(json.loads(raw))
        except (UnicodeError, ValueError):
            pass
    database = root / "runtime.db"
    if database.exists():
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for table in ("ai_state_records", "ai_state_facts"):
                for (payload,) in connection.execute(f"SELECT payload_json FROM {table}"):
                    walk(json.loads(payload))
    return b"\n".join(blobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ("image/png", "text/plain"))
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
@pytest.mark.parametrize("kind", ("none", "replay_safe", "non_replay_safe", "mcp", "read_file"))
async def test_one_result_pipeline_precedes_persistence_for_every_managed_tool(
    tmp_path: Path, kind: str, backend: str, media_type: str,
) -> None:
    principal = service_principal("default", "admitted-owner")
    state = tmp_path / "state"
    provider_counts, loads, pipeline, after_hooks, calls = [], [], [], [], []
    body = _PNG if media_type == "image/png" else _TEXT.encode()
    filename = "page.png" if media_type == "image/png" else "page.txt"
    reference = ReferencedContent(
        AssetVersionRef(AssetKey("file", "document/page-1"), "renderer",
                        StorageEntryRevision(1), hashlib.sha256(body).hexdigest(), len(body)),
        media_type=media_type, source=_SOURCE,
        representation="text" if media_type == "text/plain" else "binary",
    )
    arguments = {"path": filename} if kind == "read_file" else {"page": 1}

    class Project(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            assert context.principal == principal
            if tool.name == "finish":
                return result
            selector = (tool.metadata or {}).get(TOOL_MCP_SELECTOR_METADATA_KEY)
            assert selector == ("mcp:renderer:render" if kind == "mcp" else None)
            if kind == "read_file":
                assert tool.name == "read_file" and arguments["path"] == filename
                assert isinstance(result, ToolReturn)
                if media_type == "image/png":
                    assert result.content is None and len(result.return_value) == 2 and result.return_value[1].data == body
                else:
                    assert _TEXT in repr(result)
            else:
                assert arguments == {"page": 1}
                if media_type == "image/png":
                    assert isinstance(result, BinaryContent) and result.data == body
                else:
                    assert result == _TEXT
            assert body not in _stored_bytes(state)
            assert base64.b64encode(body) not in _stored_bytes(state)
            pipeline.append("project")
            content = reference
            if kind == "read_file" and media_type == "image/png":
                assert result.return_value[1].identifier == result.return_value[0]["files"][0]["attachment"]
                content = ReferencedContent(reference.ref, media_type=media_type, source=_SOURCE,
                                            identifier=result.return_value[1].identifier)
                return ToolReturn(return_value=[result.return_value[0], content])
            return ToolReturn(return_value={"page": 1}, content=[content])

        async def after_tool_execute(self, ctx, *, call, tool_def, args, result):
            after_hooks.append(call.tool_name)
            return result

    class Observe(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            if tool.name != "finish":
                assert pipeline == ["project"]
                assert isinstance(result, ToolReturn)
                attachment = result.return_value[1] if kind == "read_file" and media_type == "image/png" else result.content[0]
                assert attachment.ref == reference.ref
                pipeline.append("observe")
            return result

    async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        assert context.principal == principal
        assert content.ref == reference.ref and content.source == reference.source
        assert content.representation == reference.representation
        assert bool(content.identifier) is (kind == "read_file" and media_type == "image/png")
        loads.append(content)
        return body

    async def render(context: AgentContext[None], page: int) -> BinaryContent | str:
        assert context.principal == principal and page == 1
        calls.append(page)
        return BinaryContent(body, media_type=media_type) if media_type == "image/png" else _TEXT

    async def finish(context: AgentContext[None]) -> str:
        return "done"

    async def stream(messages, info):
        items = [item for message in messages for part in message.parts
                 if isinstance(part, (UserPromptPart, ToolReturnPart)) and isinstance(part.content, list)
                 for item in part.content]
        bodies = [item for item in items if isinstance(item, BinaryContent)] if media_type == "image/png" else [
            item for item in items if isinstance(item, str) and item == _TEXT
        ]
        provider_counts.append(len(bodies))
        if len(provider_counts) == 1:
            name = next(tool.name for tool in info.function_tools if tool.name != "finish")
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments), tool_call_id="render")}
        elif len(provider_counts) == 2:
            assert len(bodies) == 1
            assert (bodies[0].data if isinstance(bodies[0], BinaryContent) else bodies[0].encode()) == body
            if kind == "read_file" and media_type == "image/png":
                assert bodies[0].identifier == loads[0].identifier
            yield {0: DeltaToolCall(name="finish", json_args="{}", tool_call_id="finish")}
        else:
            assert len(provider_counts) == 3 and not bodies
            yield "done"

    features = RuntimeFeatures("projection")
    features.model("default", _Binding(stream))
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(CheckpointCapability())
    features.capability(TraceCapability(record_http_bodies=True))
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id, scope="agent")
    features.capability(CombinedCapability([Project(), Observe(), ToolResultsCapability()]),
                        id="result-projections", revision=3, scope="agent")
    features.tool(finish, effect_policy="none")
    if kind == "mcp":
        features.mcp(_server(tmp_path, text=media_type == "text/plain"))
        selected = "mcp:renderer:render"
    elif kind == "read_file":
        root = tmp_path / "workspace"
        root.mkdir()
        (root / filename).write_bytes(body)
        features.capability(WorkspaceCapability(Workspace.load(root)))
        selected = "read_file"
    else:
        features.tool(render, effect_policy=kind)
        selected = "render"
    features.agent("reader", allow_tools=(selected, "finish"), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.filesystem(state) if backend == "filesystem" else RuntimeStorage.from_root(state)
    async with Runtime.open("projection", features=features, storage=storage) as runtime:
        execution = await runtime.agents.get("reader").start("Render page one.", principal=principal)
        result = (await execution.wait(timeout_seconds=15)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        if kind == "read_file" and media_type == "image/png":
            facts = await runtime.history.attachment_facts(execution.execution_id, principal=principal)
            assert [(item.fact, item.attachment_id) for item in facts.items] == [
                ("accepted", loads[0].identifier), ("included_in_request", loads[0].identifier),
            ]
            assert all(item.source == "read_file" and item.call_id == "render" for item in facts.items)
        operations = await runtime._state.recovery.tools.list_by_execution(
            execution.execution_id, tenant_id=principal.tenant_id,
        )
        assert len(operations) == (0 if kind in {"none", "read_file"} else 1)
        if operations:
            assert operations[0].status is ToolOperationStatus.COMPLETED
            assert operations[0].replay_safe is (kind == "replay_safe")
    assert provider_counts == [0, 1, 0] and len(loads) == 1
    assert pipeline == ["project", "observe"]
    assert len(after_hooks) == 2 and after_hooks[1] == "finish"
    if kind == "mcp":
        assert (tmp_path / "calls").read_text() == "render\n"
    elif kind != "read_file":
        assert calls == [1]
    stored = _stored_bytes(state)
    assert body not in stored and base64.b64encode(body) not in stored
    assert reference.ref.digest.encode() in stored


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("none", "replay_safe", "non_replay_safe", "mcp"))
@pytest.mark.parametrize("failure", ("error", "retry", "cancel"))
async def test_projection_failure_preserves_effect_classification(
    tmp_path: Path, kind: str, failure: str,
) -> None:
    processed, calls = [], []

    class Fail(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            assert isinstance(result, BinaryContent)
            processed.append(tool.name)
            if failure == "retry":
                raise ToolCallRetry("Projection cannot establish that a tool had no effect")
            if failure == "cancel":
                raise asyncio.CancelledError()
            raise RuntimeError("Projection failed")

    async def render(context: AgentContext[None], page: int) -> BinaryContent:
        calls.append(page)
        return BinaryContent(_PNG, media_type="image/png")

    async def stream(messages, info):
        yield {0: DeltaToolCall(name=info.function_tools[0].name,
                               json_args='{"page":1}', tool_call_id="render")}

    features = RuntimeFeatures("projection-failure")
    features.model("default", _Binding(stream))
    features.capability(CheckpointCapability())
    features.capability(Fail(), id="projection", scope="agent")
    if kind == "mcp":
        features.mcp(_server(tmp_path))
        selected = "mcp:renderer:render"
    else:
        features.tool(render, effect_policy=kind)
        selected = "render"
    features.agent("reader", allow_tools=(selected,), allow_skills=(), allow_subagents=())
    state = tmp_path / "state"
    async with Runtime.open("projection-failure", features=features,
                            storage=RuntimeStorage.filesystem(state)) as runtime:
        execution = await runtime.agents.get("reader").start("Render.")
        try:
            await execution.wait(timeout_seconds=15)
        except AIError as error:
            assert kind != "none" and error.code is ErrorCode.TOOL_EFFECT_UNKNOWN
        assert len(processed) == 1
        operations = await runtime._state.recovery.tools.list_by_execution(
            execution.execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        if kind == "none":
            assert operations == ()
        else:
            assert len(operations) == 1
            assert operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
            assert operations[0].result_payload is None
            assert operations[0].replay_safe is (kind == "replay_safe")
            with pytest.raises(AIError) as raised:
                await execution.recover()
            assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    if kind == "mcp":
        assert (tmp_path / "calls").read_text() == "render\n"
    else:
        assert calls == [1]
    assert _PNG_BASE64.encode() not in _stored_bytes(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("registration", "asset_loader"))
async def test_result_capability_capture_and_restore_preserve_declared_order_and_identity(source: str) -> None:
    first, second, replacement = ToolResultsCapability(), ToolResultsCapability(), ToolResultsCapability()
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    values = (
        CapabilityContribution.from_opaque("capability", "z-first", first, revision=3),
        CapabilityContribution.from_opaque("capability", "a-second", second, revision=5),
    )

    class Loader:
        async def load(self, context: CapabilityLoadContext) -> Sequence[CapabilityContribution]:
            return values

    try:
        features = RuntimeFeatures("projection")
        if source == "registration":
            features.capability(first, id="z-first", revision=3, scope="agent")
            features.capability(second, id="a-second", revision=5, scope="agent")
        else:
            assets = AssetCapability(store)
            assets.loader("results", Loader())
            features.capability(assets)
        capture = await features.capture()
        spec = AgentSpec("reader", allow_tools=())
        models = {"default": _Binding(None)}
        compiler = AgentCompiler(models=models, candidates=capture.contributions, agents={})
        bound = compiler.bind(compiler.compile(spec), output=None)
        assert [item.value for item in bound.compiled_agent.selected_capabilities] == [first, second]
        changed = CapabilityContribution.from_opaque("capability", "z-first", replacement, revision=3)
        reopened = AgentCompiler(models=models, candidates=(changed, values[1]), agents={})
        restored = reopened.restore(bound.binding_contract)
        assert [item.value for item in restored.compiled_agent.selected_capabilities] == [replacement, second]
        mismatched = AgentCompiler(models=models, candidates=(
            CapabilityContribution.from_opaque("capability", "z-first", replacement, revision=4), values[1],
        ), agents={})
        with pytest.raises(AIError) as raised:
            mismatched.restore(bound.binding_contract)
        assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
    finally:
        await store.close()


def test_result_capability_requires_agent_binding() -> None:
    with pytest.raises(AIError) as raised:
        RuntimeFeatures().capability(ToolResultsCapability(), id="results")
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ("fixture", "completed_operation"))
async def test_final_results_bypass_live_result_projections(tmp_path: Path, source: str) -> None:
    from linktools.ai.runtime._mcp import (
        _MCPBinding, _mcp_execution_policy, _model_tool_name,
        close_mcp_resources, materialize_mcp_capabilities, prepare_mcp_projections,
    )
    from linktools.ai.runtime._tool_response_fixture import _load_tool_response_fixture
    from ._runtime_test_helpers import tool_run_context
    from .test_tool_effect_semantics import _Bridge
    from .test_tool_response_fixture import _manifest, _store

    class Reject(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            raise AssertionError("Final results must not be projected again")

    store, capabilities = None, ()
    callbacks = RunCallbacks()
    try:
        if source == "fixture":
            store, fixture = await _store(_manifest())
            manifest = await _load_tool_response_fixture(fixture, fixture.ref)
            server = MCPServerSpec("server", "unused", revision=3)
            bindings = {server.id: _MCPBinding(None, None, {"boundary": "saved-stdio"})}
            projections, tool_name = {}, "lookup"
            arguments = {"query": "ready", "options": {"a": 1, "b": 2}}
            expected = {"url": "https://example.test/info"}
            bridge = _Bridge(False)
        else:
            manifest = None
            server = _server(tmp_path)
            bindings = {server.id: _MCPBinding(None, None, _mcp_execution_policy(server, None))}
            projections = await prepare_mcp_projections(
                (server,), bindings, asset_readers={}, sandboxed=False,
            )
            tool_name, arguments = "render", {"page": 1}
            expected = ToolReturn(return_value={"page": 1}, content=[_REFERENCE])
            bridge = _Bridge(False, cached_result=expected, has_cached_result=True)
        capabilities = await materialize_mcp_capabilities(
            (server,), (f"mcp:{server.id}:{tool_name}",), sandbox=None,
            sandbox_session=None, host_cwd=str(tmp_path), bindings=bindings,
            projections=projections, tool_operations=bridge, tool_metrics=None,
            response_fixture=manifest, result_callbacks=callbacks,
        )
        boundary, context = capabilities[0].get_toolset(), tool_run_context()
        async with boundary:
            tools = await boundary.get_tools(context)
            name = _model_tool_name(server.id, tool_name)
            with callbacks.tool_scope(context.tool_call_id, Reject()):
                assert await boundary.call_tool(name, arguments, context, tools[name]) == expected
        assert bridge.calls == (["begin", "complete"] if source == "fixture" else ["begin"])
        assert not (tmp_path / "calls").exists()
    finally:
        await close_mcp_resources(capabilities)
        if store is not None:
            await store.close()


@pytest.mark.asyncio
async def test_projection_receives_validated_python_arguments_without_json_coercion() -> None:
    from datetime import datetime, timezone

    expected = datetime(2026, 1, 2, tzinfo=timezone.utc)
    calls = []

    class Observe(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            assert arguments["when"] == expected and isinstance(arguments["when"], datetime)
            with pytest.raises(TypeError):
                arguments["when"] = "changed"
            calls.append(tool.name)
            return result

    async def report(context: AgentContext[None], when: datetime) -> str:
        assert when == expected
        return "done"

    async def stream(messages, info):
        if not calls:
            yield {0: DeltaToolCall(name="report", json_args='{"when":"2026-01-02T00:00:00Z"}',
                                   tool_call_id="report")}
        else:
            yield "done"

    features = RuntimeFeatures("python-arguments")
    features.model("default", _Binding(stream))
    features.tool(report, effect_policy="none")
    features.capability(Observe(), id="observe", scope="agent")
    features.agent("reader", allow_tools=("report",), allow_subagents=(), allow_skills=())
    async with Runtime.open("python-arguments", features=features,
                            storage=RuntimeStorage.in_memory()) as runtime:
        result = (await runtime.agents.get("reader").run("Report.", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
    assert calls == ["report"]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ("none", "replay_safe", "non_replay_safe"))
@pytest.mark.parametrize("failure", ("retry", "failed", "cancel"))
async def test_original_tool_control_signals_bypass_result_processing(policy: str, failure: str) -> None:
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.capability import ToolCallFailed
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
    from ._runtime_test_helpers import tool_run_context, tool_with_metadata
    from .test_tool_effect_semantics import _Bridge

    calls = []
    error = {"retry": ToolCallRetry("retry"), "failed": ToolCallFailed("failed"),
             "cancel": asyncio.CancelledError()}[failure]

    class Reject(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            raise AssertionError("An unsuccessful tool has no successful result to project")

    async def execute() -> None:
        calls.append("execute")
        raise error

    descriptor = ManagedToolDescriptor(
        effect_owner="none" if policy == "none" else "tool_operation",
        effect_policy=policy, tool_class="business",
    )
    bridge = _Bridge(policy == "replay_safe")
    callbacks = RunCallbacks()
    boundary = BoundaryToolset(
        (FunctionToolset([tool_with_metadata(execute, descriptor)]),),
        {"execute": descriptor}, id="control-signals", tool_operations=bridge,
        result_callbacks=callbacks,
    )
    context = tool_run_context()
    tools = await boundary.get_tools(context)
    with callbacks.tool_scope(context.tool_call_id, Reject()), pytest.raises(type(error)) as raised:
        await boundary.call_tool("execute", {}, context, tools["execute"])
    assert raised.value is error
    assert calls == ["execute"]
    expected = [] if policy == "none" else ["begin", "unknown" if failure == "cancel" else "fail"]
    assert bridge.calls == expected


@pytest.mark.asyncio
async def test_authorization_precedes_tool_and_result_processing() -> None:
    from pydantic_ai.exceptions import ApprovalRequired
    from pydantic_ai.toolsets import FunctionToolset
    from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
    from linktools.ai.workspace import ToolPermissionPolicy
    from ._runtime_test_helpers import tool_run_context, tool_with_metadata
    from .test_tool_effect_semantics import _Bridge

    class Reject(ToolResultsCapability):
        async def process_tool_result(self, context, tool, arguments, result):
            raise AssertionError("An unapproved call must not reach result processing")

    async def execute() -> None:
        raise AssertionError("An unapproved call must not invoke the tool")

    descriptor = ManagedToolDescriptor(effect_owner="tool_operation",
                                       effect_policy="non_replay_safe", tool_class="filesystem.write")
    bridge = _Bridge(False)
    callbacks = RunCallbacks()
    boundary = BoundaryToolset(
        (FunctionToolset([tool_with_metadata(execute, descriptor)]),),
        {"execute": descriptor}, id="authorization", tool_operations=bridge,
        permission_policy=ToolPermissionPolicy(default="ask"), result_callbacks=callbacks,
    )
    context = tool_run_context()
    tools = await boundary.get_tools(context)
    with callbacks.tool_scope(context.tool_call_id, Reject()), pytest.raises(ApprovalRequired):
        await boundary.call_tool("execute", {}, context, tools["execute"])
    assert bridge.calls == []


@pytest.mark.asyncio
async def test_captured_execution_rebinds_result_capability_after_runtime_reopen(tmp_path: Path) -> None:
    from pydantic_ai.messages import ToolReturnPart
    from linktools.ai.runtime import AgentTaskInput, CaptureInputRequest
    from linktools.ai.task import TaskGraph, TaskNode

    calls, processed = [], []
    principal = service_principal("default", "capture-owner")

    class Project(ToolResultsCapability):
        def __init__(self, host: str) -> None:
            self.host = host

        async def process_tool_result(self, context, tool, arguments, result):
            assert context.principal == principal and tool.name == "read_text"
            assert result == _TEXT
            processed.append(self.host)
            return "Synthetic bounded summary"

    async def read_text(context: AgentContext[None]) -> str:
        calls.append(context.principal)
        return _TEXT

    async def stream(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield "done"
        else:
            yield {0: DeltaToolCall(name="read_text", json_args="{}", tool_call_id="read")}

    def features(host: str) -> RuntimeFeatures:
        result = RuntimeFeatures("reopen-projection")
        result.model("default", _Binding(stream))
        result.tool(read_text, effect_policy="replay_safe")
        result.capability(Project(host), id="result-owner", revision=3, scope="agent")
        result.agent("reader", allow_tools=("read_text",), allow_skills=(), allow_subagents=())
        return result

    async with Runtime.open("reopen-projection", features=features("original"),
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        source = await runtime.agents.get("reader").start("Inspect the synthetic document.", principal=principal)
        assert (await source.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        capture = await runtime.executions.capture_input(
            source.execution_id, CaptureInputRequest(principal, "read-capture", "clean"),
        )
    async with Runtime.open("reopen-projection", features=features("reopened"),
                            storage=RuntimeStorage.filesystem(tmp_path)) as runtime:
        task = await runtime.tasks.from_agent_capture("captured-reader", capture, principal=principal)
        graph = await runtime.tasks.bind(task).start(TaskGraph("captured-result-reader", (
            TaskNode("read", task=task, input=AgentTaskInput("Inspect the synthetic document.")),
        )), principal=principal, idempotency_key="captured-result-reader")
        await graph.wait(timeout_seconds=15)
        execution = await graph.execution("read")
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
    assert processed == ["original", "reopened"] and calls == [principal, principal]
    assert _TEXT.encode() not in _stored_bytes(tmp_path)
