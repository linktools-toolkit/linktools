#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Deferred SDK instruction state remains complete across durable history."""

import hashlib
import json
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    AgentInstructionSource, CapabilityInstructionSource, InstructionBaselinePart, InstructionDeltaPart,
    InstructionDigest, InstructionId, InstructionPart,
    ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, ToolsetInstructionSource, UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.model.messages import decode_model_messages, encode_model_messages

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


_DELTA_HAS_AUTHORITY = any(field.name == "authority" for field in fields(InstructionDeltaPart))


def _delta(*, authority: str = "operator") -> dict[str, object]:
    part = {
        "part_kind": "instruction-delta", "id": "capability:catalog:details",
        "content": "replacement", "base_hash": _HASH,
    }
    if _DELTA_HAS_AUTHORITY:
        part["authority"] = authority
    return part


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
                                  *([_delta(authority="user"), _delta(authority="none")] if _DELTA_HAS_AUTHORITY else []),
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
])
def test_instruction_window_reader_requires_declared_structure(part: dict[str, object]) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_messages(_wire(part))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("part,expected", [
    (_baseline({key: value for key, value in _block().items() if key != "name"}),
     InstructionBaselinePart(instructions=[InstructionPart("original", dynamic=True, on_change="append",
         id=InstructionId(CapabilityInstructionSource("catalog"), name="details"))])),
    (_baseline({key: value for key, value in _block(digest=True).items() if key != "dynamic"}),
     InstructionBaselinePart(instructions=[InstructionDigest(content_hash=_HASH,
         id=InstructionId(CapabilityInstructionSource("catalog"), name="details"))])),
    ({key: value for key, value in _delta().items() if key != "base_hash"},
     InstructionDeltaPart(id=InstructionId(CapabilityInstructionSource("catalog"), name="details"), content="replacement")),
])
def test_instruction_window_reader_uses_declared_defaults(part: dict[str, object], expected: object) -> None:
    restored, = decode_model_messages(_wire(part))
    assert restored == ModelRequest(parts=[expected], timestamp=_STAMP)


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


def _is_deferred_instruction_return(part: ToolReturnPart, messages: list[object]) -> bool:
    if part.content == {"instructions": "Activated instruction."}:
        return True
    if not isinstance(part.content, dict) or part.content.get("message") != (
        "Capability 'deferred' loaded. Its instructions, if any, follow."
    ):
        return False
    assert "instructions" not in part.content
    contexts = {
        f'<system><{tag} id="capability:deferred">\nActivated instruction.\n</{tag}></system>'
        for tag in ("context", "pyai:context")
    }
    return any(
        isinstance(item, UserPromptPart) and isinstance(item.content, str) and item.content in contexts
        for message in messages for item in message.parts
    )


@pytest.mark.parametrize("tag", ("context", "pyai:context"))
@pytest.mark.parametrize("identity,content,expected", (
    ("capability:deferred", "Activated instruction.", True),
    ("capability:other", "Activated instruction.", False),
    ("capability:deferred", "Different instruction.", False),
))
def test_deferred_instruction_return_requires_exact_source_and_content(
    tag: str, identity: str, content: str, expected: bool,
) -> None:
    returned = ToolReturnPart("load_capability", {
        "message": "Capability 'deferred' loaded. Its instructions, if any, follow.",
    }, tool_call_id="load-deferred")
    messages = [ModelRequest(parts=[returned, UserPromptPart(
        f'<system><{tag} id="{identity}">\n{content}\n</{tag}></system>',
    )])]
    assert _is_deferred_instruction_return(returned, messages) is expected


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
        assert any(isinstance(part, ToolReturnPart) and _is_deferred_instruction_return(part, messages)
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
    delta, = [part for part in parts if isinstance(part, InstructionDeltaPart)
              and str(part.id) == "capability:changing:details"]
    assert delta.content == "replacement" and delta.base_hash == _HASH
    assert str(delta.id) == "capability:changing:details"
    assert decode_model_messages(encode_model_messages(messages)) == tuple(messages)


async def _runtime_deferred_session(
    root: str, resumed: bool, backend: str, *, unmanaged: bool = False,
    requires_approval: bool = False, eager: bool = False,
) -> None:
    import asyncio
    from pydantic_ai.capabilities import Capability

    from linktools.ai.capability import AgentContext, CapabilityContribution, SessionCapability
    from linktools.ai.core import ApprovalDecision, ApprovalStatus, Principal, TenantAuthorizationPolicy, ToolOperationStatus
    from linktools.ai.execution import ApprovalDecisionRequest
    from linktools.ai.spec import CapabilityRef

    class HostAuthorization(TenantAuthorizationPolicy):
        pass

    state = {"text": "replacement" if resumed else "original"}
    requests, calls = 0, []

    def activated(_context: AgentContext[None]) -> str:
        calls.append("activated")
        return "native tool result"

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        names = {tool.name for tool in info.function_tools}
        if resumed or eager:
            assert "activated" in names
            if resumed:
                assert any(isinstance(part, ToolReturnPart) and _is_deferred_instruction_return(part, messages)
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
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    tool = features.tool(activated, name="activated", revision=1, effect_policy="none")
    if requires_approval:
        tool.requires_approval = True
        features._register(CapabilityContribution.from_opaque("tool", "activated", tool))
    if unmanaged:
        features.capability(_DeferredInstructions(), scope="agent")
    elif not eager:
        features.capability(
            Capability(id="deferred", defer_loading=True, instructions="Activated instruction."),
            scope="agent", tools=(CapabilityRef("tool", "activated", 1),),
        )
    features.capability(
        Capability(id="changing", instructions=_ChangingInstructions(state).get_instructions()), scope="agent", tools=(),
    )
    features.agent("default", instructions="Fixed instructions.", allow_tools=("activated",), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.sqlite(Path(root) / "runtime.db") if backend == "sqlite" else PersistenceStorage.from_root(Path(root), exclusive_writer=True)
    async with (
        Persistence.open("deferred-instructions", storage=storage) as persistence,
        Runtime.open(persistence=persistence, features=features, authorization=HostAuthorization()) as runtime,
    ):
        principal = Principal("owner", runtime.tenant_id, "service")
        sessions = runtime.sessions
        store = persistence.session_turns

        async def committed_messages():
            row = await runtime.history.turn_session_state("session", principal=principal)
            turns = await store.read_context_range("session", after=0, through=row["last_turn_no"])
            return tuple(message for turn in turns for message in turn.messages)

        if resumed:
            assert encode_model_messages(await committed_messages()) == (Path(root) / "expected.json").read_bytes()
        else:
            await sessions.create_turn_session("session", principal=principal)
        if unmanaged:
            with pytest.raises(AIError) as rejected:
                await sessions.start("session", "Load capability.", agent="default", principal=principal,
                                     idempotency_key="unmanaged")
            assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID
            assert rejected.value.safe_details["reason"] == "turn_agent_features_unsupported"
            assert not requests and not calls
            assert (await runtime.history.turn_session_state("session", principal=principal))["active_turn_id"] is None
            return
        turn = await sessions.start("session", "Continue." if resumed else "Load capability.",
                                    agent="default", principal=principal, idempotency_key="resume" if resumed else "load")
        if requires_approval and (resumed or eager):
            waiting = await asyncio.wait_for(turn.wait(), 10)
            assert waiting.status is ExecutionStatus.WAITING_DEFERRED, waiting.view
            [approval] = await runtime.approvals.list(turn.key, principal=principal)
            assert approval.tool_name == "activated" and approval.status is ApprovalStatus.PENDING
            assert approval.arguments == {} and not calls
            pending = await store.get_waiting_turn(turn.key)
            assert pending is not None and pending.key == turn.key
            [operation_id] = pending.approval_operation_ids
            assert not pending.external_operation_ids
            tools = persistence.checkpoint.tools
            operation = await tools.get_operation(operation_id, tenant_id=principal.tenant_id)
            assert operation is not None and operation.status is ToolOperationStatus.PENDING
            assert operation.execution_id == turn.key.turn_id and operation.tool_call_id == "use-activated"
            assert operation.capability_ref == CapabilityRef("tool", "activated", 1)
            receipt = await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
                principal, approval.approval_id, "approve-activated", ApprovalDecision.APPROVE,
            ))
            assert receipt.approval_id == approval.approval_id and receipt.decision is ApprovalDecision.APPROVE
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED, result.view
        assert result.view["output"] == {"text": "done"} and requests == 2
        assert calls == (["activated"] if resumed or eager else [])
        if requires_approval and (resumed or eager):
            settled = await tools.list_by_execution(turn.key.turn_id, tenant_id=principal.tenant_id)
            [completed] = [entry for entry in settled if entry.tool_name == "activated"]
            assert completed.tool_operation_id == operation_id and completed.status is ToolOperationStatus.COMPLETED
            assert completed.capability_ref == operation.capability_ref
            repeated = await sessions.start("session", "Continue." if resumed else "Load capability.",
                                            agent="default", principal=principal,
                                            idempotency_key="resume" if resumed else "load")
            assert repeated.key == turn.key and (await repeated.result()).view == result.view
            assert calls == ["activated"] and requests == 2
        if eager:
            return
        messages = await committed_messages()
        parts = [part for message in messages for part in message.parts]
        assert any(isinstance(part, InstructionBaselinePart) for part in parts)
        assert any(isinstance(part, InstructionDeltaPart) and part.content == "replacement" for part in parts)
        if not resumed:
            (Path(root) / "expected.json").write_bytes(encode_model_messages(messages))


@pytest.mark.parametrize("backend", ["local_sqlite", "sqlite"])
@pytest.mark.parametrize("requires_approval", (False, True))
def test_runtime_fresh_process_reopen_retains_instructions_and_deferred_activation(
    tmp_path: Path, backend: str, requires_approval: bool,
) -> None:
    script = ("import asyncio, sys; from tests.ai.test_instruction_message_contract import _runtime_deferred_session; "
              "asyncio.run(_runtime_deferred_session(sys.argv[1], sys.argv[2] == 'reopen', sys.argv[3], requires_approval=sys.argv[4] == 'True'))")
    for phase in ("write", "reopen"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), phase, backend, str(requires_approval)],
                                capture_output=True, text=True, timeout=35)
        assert result.returncode == 0, result.stdout + result.stderr



@pytest.mark.asyncio
async def test_native_session_rejects_unmanaged_deferred_function_toolset(tmp_path: Path) -> None:
    await _runtime_deferred_session(str(tmp_path), False, "sqlite", unmanaged=True)



@pytest.mark.asyncio
async def test_native_session_registered_eager_tool_approval_has_one_effect(tmp_path: Path) -> None:
    await _runtime_deferred_session(str(tmp_path), False, "sqlite", requires_approval=True, eager=True)
