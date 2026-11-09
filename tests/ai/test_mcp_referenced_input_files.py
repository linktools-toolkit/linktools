#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP originals borrow the existing process/resource lifetime and stable receipts."""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.asset import AssetMaterializer
from linktools.ai.capability import ExecutionCapability, ReferencedInputResult, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._tool_boundary import BoundaryToolset, ManagedToolDescriptor
from linktools.ai.spec import MCPServerSpec, mcp_tool_selector

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, tool_run_context, tool_with_metadata
from .test_referenced_content import _BODY, _SOURCE, _content
from .test_tool_effect_semantics import _Bridge

_SELECTOR = mcp_tool_selector("original-reader", "inspect_files")


def _server(tmp_path: Path, mode: str) -> MCPServerSpec:
    script = tmp_path / "reader.py"
    script.write_text(f'''
import hashlib, json, os, signal, sys, time
from pathlib import Path
root = Path({str(tmp_path)!r})
paths = []
def exiting(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "exited.tmp").write_text(json.dumps([path.is_file() for path in paths]))
    (root / "exited.tmp").replace(root / "exited")
    sys.exit(0)
signal.signal(signal.SIGTERM, exiting)
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {{"protocolVersion": request["params"]["protocolVersion"], "capabilities": {{"tools": {{}}}},
                  "serverInfo": {{"name": "original-reader", "version": "1"}}}}
    elif method == "tools/list":
        result = {{"tools": [{{"name": "inspect_files", "description": "Inspect two admitted original tokens.",
                  "inputSchema": {{"type": "object", "properties": {{"left": {{"type": "string"}}, "right": {{"type": "string"}}}},
                  "required": ["left", "right"], "additionalProperties": False}}}}]}}
    elif method == "tools/call":
        args = request["params"]["arguments"]
        paths = [Path(args["left"]), Path(args["right"])]
        assert all(path.is_absolute() and path.is_file() for path in paths)
        (root / "called.tmp").write_text(json.dumps({{"pid": os.getpid(), "paths": [str(path) for path in paths],
            "digests": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]}}))
        (root / "called.tmp").replace(root / "called")
        if {mode!r} == "cancel":
            while True:
                time.sleep(0.1)
        result = {{"content": [{{"type": "text", "text": "verified originals"}}], "isError": {mode == 'failure'!r}}}
    elif method == "ping":
        result = {{}}
    else:
        continue
    print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}), flush=True)
exiting(None, None)
''', encoding="utf-8")
    return MCPServerSpec("original-reader", sys.executable, (str(script),))


async def _wait_file(path: Path) -> None:
    async with asyncio.timeout(30):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["success", "revoked", "failure", "cancel", "load_cancel"])
async def test_mcp_reads_admitted_files_without_changing_receipts_and_cleans_after_exit(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _content()
    arguments = {"left": "admitted-upload", "right": "admitted-upload"}
    authorized = True
    resolver_calls, batch_calls = [], []
    loading = asyncio.Event()
    requests = 0
    cleanup_checks = []
    original_close = AssetMaterializer.close

    async def close_after_process(materializer):
        called = tmp_path / "called"
        if called.exists():
            observed = json.loads(called.read_text())
            if all(Path(path).exists() for path in observed["paths"]):
                # Forced cancellation may use SIGKILL, so a child exit handler
                # cannot prove ordering. Inspect the real process before removal.
                if os.name == "posix":
                    with pytest.raises(ProcessLookupError):
                        os.kill(observed["pid"], 0)
                cleanup_checks.append(observed["paths"])
        await original_close(materializer)

    monkeypatch.setattr(AssetMaterializer, "close", close_after_process)

    async def resolve_inputs(context, args):
        resolver_calls.append(context)
        assert context.tool_call.tool_call_id == "inspect"
        assert args == arguments
        with pytest.raises(TypeError):
            args["left"] = "mutated"
        return {"left": reference, "right": reference}

    async def load(context, contents):
        if context.tool_call is not None:
            batch_calls.append((context, contents))
            if mode == "load_cancel":
                loading.set()
                await asyncio.Event().wait()
        if not authorized:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        assert all(value.ref == reference.ref and value.media_type == reference.media_type for value in contents)
        return [ReferencedInputResult(value, _BODY) for value in contents]

    async def respond(messages, info):
        nonlocal requests, authorized
        requests += 1
        if requests == 1:
            if mode == "revoked":
                authorized = False
            return ModelResponse(parts=[ToolCallPart(info.function_tools[0].name, arguments, tool_call_id="inspect")])
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("mcp-original-inputs")
    group.mcp(_server(tmp_path, mode))
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    group.capability(TraceCapability(), scope="runtime")
    group.capability(ReferencedInputsCapability(authorized_load_many=load, mcp_inputs={_SELECTOR: resolve_inputs}),
                     id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    group.agent("default", allow_tools=(_SELECTOR,), allow_skills=(), allow_subagents=())
    models = _Models(_UsageFunctionModel(respond))
    models.contract = {**models.contract, "input_mime_types": ["application/pdf"]}
    storage = PersistenceStorage.from_root(tmp_path / "state", exclusive_writer=True)
    async with Persistence.open("mcp-original-inputs", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(models, group),
    ) as runtime:
        execution = await runtime.agents.get().start(["Inspect the admitted original", reference])
        waiting = asyncio.create_task(execution.wait(timeout_seconds=45))
        if mode == "cancel":
            await _wait_file(tmp_path / "called")
            observed = json.loads((tmp_path / "called").read_text())
            assert all(Path(path).is_file() for path in observed["paths"])
            await execution.cancel()
        elif mode == "load_cancel":
            await asyncio.wait_for(loading.wait(), timeout=30)
            await execution.cancel()
        if mode in {"failure", "cancel"}:
            with pytest.raises(AIError) as unknown:
                await waiting
            assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            state = await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)
            assert state.status is ExecutionStatus.RECOVERY_REQUIRED
        else:
            result = (await waiting).result
        if mode == "revoked":
            assert result.status is ExecutionStatus.FAILED
            assert not (tmp_path / "called").exists()
        elif mode == "load_cancel":
            assert result.status is ExecutionStatus.CANCELLED
            assert not (tmp_path / "called").exists()
        elif mode == "success":
            assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        records = await runtime._state.checkpoint.tools.list_by_execution(
            execution.execution_id, tenant_id=runtime.default_principal.tenant_id,
        )
        assert len(records) == 1 and records[0].arguments_digest == canonical_sha256(arguments)
        if mode in {"revoked", "load_cancel"}:
            assert records[0].status is not ToolOperationStatus.EFFECT_UNKNOWN
        elif mode in {"failure", "cancel"}:
            assert records[0].status is ToolOperationStatus.EFFECT_UNKNOWN
        assert len(batch_calls) == len(resolver_calls) == 1
        assert batch_calls[0][0].execution_id == execution.execution_id
        assert len(batch_calls[0][1]) == 2
    if mode not in {"revoked", "load_cancel"}:
        observed = json.loads((tmp_path / "called").read_text())
        assert observed["digests"] == [hashlib.sha256(_BODY).hexdigest()] * 2
        assert cleanup_checks == [observed["paths"]]
        assert all(not Path(path).exists() for path in observed["paths"])
        stored = b"\n".join(path.read_bytes() for path in (tmp_path / "state").rglob("*") if path.is_file())
        assert not any(path.encode() in stored for path in observed["paths"])


@pytest.mark.asyncio
async def test_completed_call_does_not_resolve_or_materialize_inputs_again() -> None:
    async def invoke(path: str):
        pytest.fail("A completed operation cannot invoke the tool again")

    async def prepare(*args):
        pytest.fail("Cached results do not resolve or read revoked input")

    descriptor = ManagedToolDescriptor("tool_operation", "replay_safe", "business")
    bridge = _Bridge(True, cached_result="completed", has_cached_result=True)
    boundary = BoundaryToolset((FunctionToolset([tool_with_metadata(invoke, descriptor)]),),
                               {"invoke": descriptor}, id="cached-original",
                               tool_operations=bridge, prepare_arguments=prepare)
    context = tool_run_context()
    tools = await boundary.get_tools(context)
    assert await boundary.call_tool("invoke", {"path": "admitted-upload"}, context, tools["invoke"]) == "completed"
    assert bridge.calls == ["begin"]


@pytest.mark.asyncio
async def test_verified_bytes_projection_reuses_integrity_path_and_cleanup_contract() -> None:
    reference = _content().ref
    async with AssetMaterializer() as materializer:
        with pytest.raises(AIError) as invalid:
            await materializer.materialize_bytes({"original": (reference, b"changed")})
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(ValueError):
            await materializer.materialize_bytes({"../escape": (reference, _BODY)})
        value = await materializer.materialize_bytes({"original": (reference, _BODY)})
        assert value.files["original"].read_bytes() == _BODY
        assert value.files["original"].stat().st_mode & 0o111 == 0
    assert not value.root.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["remote", "field", "source", "conflict", "budget"])
async def test_mcp_file_handoff_rejects_unreachable_invalid_or_oversized_inputs_before_reading(failure: str) -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from linktools.ai.core import PromptLimits
    from linktools.ai.model import ReferencedContent
    from linktools.ai.runtime._mcp import _MCPModelToolset
    from linktools.ai.spec import CapabilityRef
    from .test_referenced_content import _context

    second_source = CapabilityRef("capability", "second-files", 1)
    reference = _content()
    second = ReferencedContent(reference.ref, media_type=reference.media_type, source=second_source)

    async def load(context, contents):
        pytest.fail("Rejected inputs must not authorize or read any source")

    async def resolve(context, arguments):
        if failure == "remote":
            pytest.fail("Remote servers must not resolve local paths")
        return {"missing" if failure == "field" else "left": second if failure == "source" else reference}

    async def resolve_second(context, arguments):
        return {"left" if failure == "conflict" else "right": second}

    async def inspect_files(left: str, right: str) -> str:
        pytest.fail("Rejected handoffs must not invoke the leaf")

    sources = {_SOURCE: ReferencedInputsCapability(authorized_load_many=load, mcp_inputs={_SELECTOR: resolve})}
    if failure in {"conflict", "budget"}:
        sources[second_source] = ReferencedInputsCapability(authorized_load_many=load,
                                                          mcp_inputs={_SELECTOR: resolve_second})
    async with AssetMaterializer() as materializer:
        mapped = _MCPModelToolset(FunctionToolset([inspect_files]), "original-reader", None,
                                 input_callbacks=SimpleNamespace(referenced_inputs=lambda: sources),
                                 materializer=materializer, local_input_paths=failure != "remote",
                                 input_limits=PromptLimits(max_binary_input_bytes=len(_BODY)))
        context = replace(tool_run_context(), deps=_context())
        name, tool = next(iter((await mapped.get_tools(context)).items()))
        context = replace(context, tool_name=name)
        stable = {"left": "admitted-upload", "right": "admitted-upload"}
        with pytest.raises(AIError) as rejected:
            await mapped.prepare_arguments(name, stable, context, tool)
        assert rejected.value.code is {
            "remote": ErrorCode.CAPABILITY_RESOLUTION_INVALID,
            "field": ErrorCode.REQUEST_FIELD_INVALID,
            "source": ErrorCode.CAPABILITY_RESOLUTION_INVALID,
            "conflict": ErrorCode.CAPABILITY_CONFLICT,
            "budget": ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE,
        }[failure]
        assert stable == {"left": "admitted-upload", "right": "admitted-upload"}


def test_mcp_input_mapping_requires_exact_formal_selector() -> None:
    async def unused(*args):
        pytest.fail("Declaration validation does not call the host")

    for selector in ("mcp:original-reader:*", "inspect_files"):
        with pytest.raises(ValueError):
            ReferencedInputsCapability(authorized_load_many=unused, mcp_inputs={selector: unused})
