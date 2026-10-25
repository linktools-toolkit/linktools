#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred SDK instruction state remains complete across durable history."""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentInstructionSource, CapabilityInstructionSource, InstructionBaselinePart, InstructionDeltaPart,
    InstructionDigest, InstructionId, InstructionPart,
    ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, ToolsetInstructionSource,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeStorage
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages

_STAMP = datetime(2026, 10, 24, tzinfo=timezone.utc)
_HASH = hashlib.sha256(b"original").hexdigest()


def _wire(part: dict[str, object]) -> bytes:
    value = json.loads(encode_model_messages((ModelRequest(parts=[], timestamp=_STAMP),)))
    value[0]["parts"] = [part]
    return canonical_json_bytes(value)


def _block(*, digest: bool = False) -> dict[str, object]:
    common = {"id": "capability:catalog:details", "dynamic": True}
    return ({**common, "part_kind": "instruction-digest", "content_hash": _HASH} if digest else
            {**common, "part_kind": "instruction", "content": "original", "on_change": "append", "name": "details"})


def _baseline(block: object) -> dict[str, object]:
    return {"part_kind": "instruction-baseline", "instructions": [block]}


def _delta() -> dict[str, object]:
    return {"part_kind": "instruction-delta", "id": "capability:catalog:details", "content": "replacement", "base_hash": _HASH}


@pytest.mark.parametrize("source", [None, AgentInstructionSource(), ToolsetInstructionSource("tools"),
                                    CapabilityInstructionSource("catalog")])
@pytest.mark.parametrize("name", [None, "details"])
def test_instruction_window_round_trip_preserves_order_identity_and_termination(source: object, name: str | None) -> None:
    identity = None if source is None else InstructionId(source, name=name)
    full = InstructionPart("original", id=identity, name=name, dynamic=True, on_change="append")
    digest = InstructionDigest.of(InstructionPart("fixed", id=identity, dynamic=False))
    anonymous = InstructionPart("", id=None, dynamic=False, on_change="rewrite", name=None)
    message = ModelRequest(parts=[
        InstructionBaselinePart(instructions=[digest, full, anonymous]),
        InstructionDeltaPart(id=identity, content="replacement", base_hash=_HASH),
        InstructionDeltaPart(id=identity, content=None, base_hash=None),
        InstructionBaselinePart(instructions=[]),
    ], timestamp=_STAMP)
    encoded = encode_model_messages((message,))
    assert json.loads(encoded)[0]["version"] == 1
    restored, = decode_model_messages(encoded)
    assert restored == message
    assert [type(block) for block in restored.parts[0].instructions] == [InstructionDigest, InstructionPart, InstructionPart]
    assert restored.parts[2].content is None and restored.parts[2].base_hash is None
    assert restored.parts[3].instructions == []
    assert encode_model_messages((restored,)) == encoded


@pytest.mark.parametrize("part", [_baseline(_block()), _baseline(_block(digest=True)), _delta(),
                                  {"part_kind": "instruction-baseline", "instructions": []}])
def test_instruction_window_reader_preserves_supported_parts(part: dict[str, object]) -> None:
    restored, = decode_model_messages(_wire(part))
    assert json.loads(encode_model_messages((restored,)))[0]["parts"] == [part]


@pytest.mark.parametrize("digest,field,value", [
    (False, "content", None), (False, "dynamic", 1), (False, "on_change", "unknown"),
    (False, "name", []), (False, "id", {}), (True, "dynamic", "false"),
    (True, "content_hash", None), (True, "content_hash", "not-a-sha256"), (True, "id", 2),
])
def test_instruction_baseline_reader_rejects_malformed_known_fields(digest: bool, field: str, value: object) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(_baseline({**_block(digest=digest), field: value})))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("field,value", [("content", 7), ("base_hash", False), ("base_hash", "g" * 64), ("id", [])])
def test_instruction_delta_reader_rejects_malformed_known_fields(field: str, value: object) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire({**_delta(), field: value}))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("part", [
    {"part_kind": "instruction-baseline", "instructions": None}, _baseline(None),
    _baseline({key: value for key, value in _block().items() if key != "name"}),
    _baseline({key: value for key, value in _block(digest=True).items() if key != "dynamic"}),
    {key: value for key, value in _delta().items() if key != "base_hash"},
])
def test_instruction_window_reader_requires_complete_shapes(part: dict[str, object]) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(part))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("part", [
    _baseline({**_block(), "part_kind": "future-instruction"}),
    _baseline({**_block(), "id": "future:catalog"}),
    _baseline({**_block(digest=True), "id": "future:catalog"}),
    {**_delta(), "id": "future:catalog"},
])
def test_instruction_reader_does_not_erase_unknown_blocks_or_source_namespaces(part: dict[str, object]) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(part))
    assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.parametrize("invalid", ["unknown-block", "dynamic", "policy", "hash", "id"])
def test_instruction_writer_rejects_invalid_values_instead_of_dropping_them(invalid: str) -> None:
    block = InstructionPart("original", id=InstructionId(AgentInstructionSource()), on_change="append")
    if invalid == "unknown-block":
        block = object()
    elif invalid == "hash":
        block = InstructionDigest(content_hash="bad")
    elif invalid == "id":
        block.id = "agent"
    elif invalid == "dynamic":
        block.dynamic = 1
    else:
        block.on_change = "unknown"
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((ModelRequest(parts=[InstructionBaselinePart(instructions=[block])]),))


class _DeferredInstructions(AbstractCapability):
    id, defer_loading = "deferred", True

    def get_instructions(self) -> str:
        return "Activated instruction."

    def get_toolset(self) -> FunctionToolset:
        def activated() -> str:
            return "native tool result"
        return FunctionToolset([activated])


class _ChangingInstructions(AbstractCapability):
    id = "changing"

    def __init__(self, state: dict[str, str]) -> None:
        self.state = state

    def get_instructions(self) -> Callable[[], InstructionPart]:
        def instructions() -> InstructionPart:
            return InstructionPart(self.state["text"], name="details", on_change="append")
        return instructions


@pytest.mark.asyncio
async def test_native_deferred_two_request_writer_round_trips_real_baseline_digest_and_delta() -> None:
    state = {"text": "original"}
    requests = []

    def model(messages, info) -> ModelResponse:
        requests.append(messages)
        if len(requests) == 1:
            assert "activated" not in {tool.name for tool in info.function_tools}
            state["text"] = "replacement"
            return ModelResponse(parts=[ToolCallPart("load_capability", {"id": "deferred"}, "load-deferred")])
        assert len(requests) == 2
        assert "activated" in {tool.name for tool in info.function_tools}
        assert any(isinstance(part, ToolReturnPart) and part.content == {"instructions": "Activated instruction."}
                   for message in messages for part in message.parts)
        return ModelResponse(parts=[TextPart("done")])

    result = await Agent(FunctionModel(model), instructions="Fixed instructions.", capabilities=[
        _DeferredInstructions(), _ChangingInstructions(state),
    ]).run("Load the deferred capability.")
    assert result.output == "done" and len(requests) == 2
    messages = result.all_messages()
    parts = [part for message in messages for part in message.parts]
    baseline, = [part for part in parts if isinstance(part, InstructionBaselinePart)]
    assert any(isinstance(block, InstructionDigest) for block in baseline.instructions)
    assert any(isinstance(block, InstructionPart) for block in baseline.instructions)
    delta, = [part for part in parts if isinstance(part, InstructionDeltaPart)]
    assert delta.content == "replacement" and delta.base_hash == _HASH
    assert str(delta.id) == "capability:changing:details"
    assert decode_model_messages(encode_model_messages(messages)) == tuple(messages)


async def _runtime_deferred_session(root: str, resumed: bool, backend: str) -> None:
    state = {"text": "replacement" if resumed else "original"}
    requests = 0

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        names = {tool.name for tool in info.function_tools}
        if resumed:
            assert "activated" in names
            assert any(isinstance(part, ToolReturnPart) and part.content == {"instructions": "Activated instruction."}
                       for message in messages for part in message.parts)
            if requests == 1:
                yield {0: DeltaToolCall(name="activated", json_args="{}", tool_call_id="use-activated")}
                return
            assert any(isinstance(part, ToolReturnPart) and part.content == "native tool result"
                       for message in messages for part in message.parts)
        elif requests == 1:
            assert "activated" not in names
            state["text"] = "replacement"
            yield {0: DeltaToolCall(name="load_capability", json_args='{"id":"deferred"}', tool_call_id="load-deferred")}
            return
        assert "activated" in names and requests == 2
        yield "done"

    class Binding:
        provider, model_identity = "test", "test:deferred-instructions"
        contract = {}

        def materialize(self) -> FunctionModel:
            return FunctionModel(stream_function=stream)

    features = RuntimeFeatures("deferred-instructions")
    features.model("default", Binding())
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(_DeferredInstructions(), scope="agent")
    features.capability(_ChangingInstructions(state), scope="agent")
    features.agent("default", instructions="Fixed instructions.", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.sqlite(Path(root) / "runtime.db") if backend == "sqlite" else RuntimeStorage.filesystem(Path(root))
    async with Runtime.open("deferred-instructions", features=features, storage=storage) as runtime:
        agent = runtime.agents.get()
        session = agent.session("session") if resumed else await agent.create_session("session")
        if resumed:
            messages = (await runtime.history.messages("session", principal=runtime.default_principal)).items
            assert encode_model_messages(messages) == (Path(root) / "expected.json").read_bytes()
            context = await runtime._session_service.load_model_context("session", principal=runtime.default_principal)
            assert encode_model_messages(context) == (Path(root) / "expected-context.json").read_bytes()
        result = (await session.run("Continue." if resumed else "Load capability.", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED, result.error_diagnostics
        assert result.output == {"text": "done"} and requests == 2
        messages = (await runtime.history.messages("session", principal=runtime.default_principal)).items
        # SDK instruction windows belong to the persisted working context, not the raw transcript.
        context = await runtime._session_service.load_model_context("session", principal=runtime.default_principal)
        parts = [part for message in context for part in message.parts]
        assert any(isinstance(part, InstructionBaselinePart) for part in parts)
        assert any(isinstance(part, InstructionDeltaPart) and part.content == "replacement" for part in parts)
        if not resumed:
            (Path(root) / "expected.json").write_bytes(encode_model_messages(messages))
            (Path(root) / "expected-context.json").write_bytes(encode_model_messages(context))


@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
def test_runtime_fresh_process_reopen_retains_instructions_and_deferred_activation(tmp_path: Path, backend: str) -> None:
    script = ("import asyncio, sys; from tests.ai.test_instruction_message_contract import _runtime_deferred_session; "
              "asyncio.run(_runtime_deferred_session(sys.argv[1], sys.argv[2] == 'reopen', sys.argv[3]))")
    for phase in ("write", "reopen"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), phase, backend],
                                capture_output=True, text=True, timeout=35)
        assert result.returncode == 0, result.stdout + result.stderr
