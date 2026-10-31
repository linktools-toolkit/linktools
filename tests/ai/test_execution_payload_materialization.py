#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending execution payloads remain self-contained until a source boundary."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain, PromptLimits, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext, RuntimePayloadRef
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.persistence.objects import ObjectKeyFactory, read_runtime_object
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._execution_payload import _materialize_execution_payloads
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, PayloadPolicy, StoredPayload

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_history_projection_conformance import _record


class _Objects(InMemoryObjectStore):
    def __init__(self):
        super().__init__()
        self.writes = []

    async def put(self, key, chunks, *, expected_size, expected_digest):
        result = await super().put(key, chunks, expected_size=expected_size, expected_digest=expected_digest)
        self.writes.append(result)
        return result


@pytest.mark.asyncio
async def test_captured_input_is_self_contained_before_object_materialization() -> None:
    objects = _Objects()
    materializer = ExecutionInputMaterializer(
        None, PromptLimits(), object_store=objects,
        object_key_factory=ObjectKeyFactory("capture"), payload_policy=PayloadPolicy(inline_limit_bytes=128),
    )
    prompt = ("Inspect", BinaryContent(b"body" * 100, media_type="text/plain"))
    captured = materializer.capture(prompt)
    encoded = encode_domain(captured)
    restored = decode_domain(encoded, type(captured))
    assert objects.writes == []
    assert tuple(await materializer.restore(restored)) == prompt
    placed = await materializer.materialize_payload(restored, tenant_id="tenant")
    assert placed.payload.kind == "object"
    assert placed.payload.digest == captured.payload.digest
    assert tuple(await materializer.restore(placed)) == prompt
    assert await materializer.materialize_payload(placed, tenant_id="tenant") is placed
    assert len(objects.writes) == 1
    assert captured.payload.kind == "inline"


@pytest.mark.asyncio
async def test_instruction_placement_uses_frozen_bytes_without_mutating_capture() -> None:
    objects = _Objects()
    instructions = RepositoryInstructions((
        RepositoryInstructionDocument("agents:AGENTS.md", ".", "rule" * 100),
    ))
    payload = StoredPayload.inline_json(instructions.to_payload())
    captured = replace(_record(ExecutionStatus.PENDING_START, 0),
                       repository_instructions=RuntimePayloadRef(payload, PersistenceDomain.EXECUTION))
    placed = await _materialize_execution_payloads(
        captured, input_materializer=None, objects=objects, keys=ObjectKeyFactory("capture"),
        policy=PayloadPolicy(inline_limit_bytes=128), tenant_id="tenant",
    )
    reference = placed.repository_instructions.payload
    assert reference.kind == "object"
    assert reference.digest == payload.digest
    assert await read_runtime_object(objects, reference.ref) == canonical_json_bytes(instructions.to_payload())
    assert captured.repository_instructions.payload is payload
    assert captured.repository_instructions.payload.kind == "inline"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "terminal", "promotion"))
async def test_large_pending_input_is_materialized_only_at_source_boundary(tmp_path: Path, mode: str) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    objects = _Objects()
    prompt = ("input" * 16000,)
    answer = "output" * 13000

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart(answer)])

    features = RuntimeFeatures("payload-placement")
    features.capability(ExecutionCapability(
        retain_history=True, commit=CommitPolicy.sync() if mode == "sync" else CommitPolicy.terminal(),
    ), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(
        tmp_path / "state.db", object_store=objects,
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Persistence.open("payload-placement", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start(prompt, idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            source = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            if mode == "sync":
                assert source is not None
                assert source.stored_user_input.payload.kind == "object"
                assert objects.writes
            else:
                assert source is None
                assert objects.writes == []
            if mode == "promotion":
                await runtime._execution_service.runtime_backend().promote_pending_execution(execution.execution_id)
                source = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert source is not None
                assert source.stored_user_input.payload.kind == "object"
                assert tuple(await runtime._execution_service.runtime_backend().restore_user_input(source)) == prompt
        finally:
            release.set()
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.output == {"text": answer}
        source = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert source is not None
        assert source.stored_user_input.payload.kind == "object"
        assert tuple(await runtime._execution_service.runtime_backend().restore_user_input(source)) == prompt
        replay = await session.start(prompt, idempotency_key="first")
        assert replay.execution_id == execution.execution_id
        assert (await replay.wait()).result.output == result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("imported", (False, True))
async def test_rejected_admission_does_not_publish_large_input_or_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, imported: bool,
) -> None:
    objects = _Objects()
    features = RuntimeFeatures("rejected-payload")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    context = ExecutionInputContext.from_messages((
        ModelRequest(parts=[UserPromptPart("history" * 12000)]),
    )) if imported else None

    async def model(messages, info):
        raise AssertionError("rejected admission must not start the provider")

    async def reject(*args, **kwargs):
        return None

    storage = PersistenceStorage.sqlite(
        tmp_path / "state.db", object_store=objects,
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with Persistence.open("rejected-payload", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        coordination = runtime._execution_service._runtime_bridge.pending_turns.coordination
        monkeypatch.setattr(type(coordination), "claim", reject)
        with pytest.raises(AIError) as raised:
            await runtime.agents.get().start(("input" * 16000,), input_context=context)
        assert raised.value.code is ErrorCode.SESSION_BUSY
        assert objects.writes == []


@pytest.mark.asyncio
async def test_imported_context_is_archived_at_source_and_can_reopen(tmp_path: Path) -> None:
    objects = _Objects()
    context = ExecutionInputContext.from_messages((
        ModelRequest(parts=[UserPromptPart("history" * 12000)]),
        ModelResponse(parts=[TextPart("prior answer")]),
    ), session_metadata={"large": "metadata" * 10000})
    features = RuntimeFeatures("imported-payload")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())

    async def model(messages, info):
        return ModelResponse(parts=[TextPart("answer")])

    def storage():
        return PersistenceStorage.sqlite(tmp_path / "state.db", object_store=objects,
            coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))

    async with Persistence.open("imported-payload", storage=storage()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        execution = await runtime.agents.get().start("question", input_context=context)
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        reference = record.input_context
        assert reference.payload.decode()["kind"] == "execution-context"
        assert objects.writes
    async with Persistence.open("imported-payload", storage=storage()) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        restored = await runtime._execution_service.runtime_backend().load_input_context(reference)
        assert restored == context
