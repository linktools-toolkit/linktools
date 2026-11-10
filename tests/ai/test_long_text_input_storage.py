#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Large text prompts share immutable payload bytes across retained turns."""

from dataclasses import replace

import pytest

from linktools.ai.core import PersistenceDomain, PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history._session_timeline import load_stored_input_view
from linktools.ai.inputs import ExecutionInputMaterializer, stored_user_input_view
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence.objects import ObjectKeyFactory, read_runtime_object
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, PayloadPolicy, StoredPayload


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ('a"\\\n' * 20000, "中文🙂" * 10000))
async def test_large_text_keeps_identity_without_inline_display_duplicate(text: str) -> None:
    objects = InMemoryObjectStore()
    materializer = ExecutionInputMaterializer(
        None, PromptLimits(), object_store=objects,
        object_key_factory=ObjectKeyFactory("large-input"), payload_policy=PayloadPolicy(64),
    )
    captured = materializer.capture(text)
    stored = await materializer.materialize_payload(captured, tenant_id="tenant")
    assert stored.payload.kind == "object"
    assert stored.view is None
    assert stored.digest == captured.digest
    assert stored.payload.digest == captured.payload.digest
    assert stored.payload.size == len(text.encode("utf-8"))
    assert await materializer.restore(stored) == text
    assert await materializer.materialize_payload(stored, tenant_id="tenant") is stored

    async def read(payload: StoredPayload) -> bytes:
        assert payload.ref is not None
        return await read_runtime_object(objects, payload.ref)

    assert await load_stored_input_view(stored, read) == stored_user_input_view(captured)


@pytest.mark.asyncio
async def test_retained_text_survives_execution_object_cleanup() -> None:
    execution_objects, conversation_objects = InMemoryObjectStore(), InMemoryObjectStore()
    materializer = ExecutionInputMaterializer(
        None, PromptLimits(), object_store=execution_objects,
        object_key_factory=ObjectKeyFactory("large-input"), payload_policy=PayloadPolicy(16),
    )
    text = "retained prompt " * 10000
    stored = await materializer.store(text, tenant_id="tenant")

    async def read(domain: PersistenceDomain, reference: ObjectRef) -> bytes:
        objects = execution_objects if domain is PersistenceDomain.EXECUTION else conversation_objects
        return await read_runtime_object(objects, reference)

    history = TranscriptRepository(
        InMemoryStateStore(), object_store=conversation_objects,
        namespace="large-input", tenant_id="tenant", runtime_domain=PersistenceDomain.CONVERSATION,
        object_reader=read,
    )
    assert await history.read_payload(stored.payload, source_domain=PersistenceDomain.EXECUTION) == text.encode()
    retained = replace(stored, payload=await history.retain_payload(
        stored.payload, source_domain=PersistenceDomain.EXECUTION,
    ))
    assert retained.payload == stored.payload
    assert stored.payload.ref is not None
    await execution_objects.delete_object(stored.payload.ref.key, expected_digest=stored.payload.digest)
    assert await load_stored_input_view(retained, history.read_payload) == stored_user_input_view(materializer.capture(text))


@pytest.mark.asyncio
async def test_text_view_checks_object_bytes_and_small_input_stays_inline() -> None:
    materializer = ExecutionInputMaterializer(None, PromptLimits(), payload_policy=PayloadPolicy(256))
    captured = materializer.capture("small")
    assert await materializer.materialize_payload(captured, tenant_id="tenant") is captured
    reference = ObjectRef("runtime", "test-key", captured.payload.digest, captured.payload.size)
    stored = replace(captured, payload=StoredPayload.object(reference))

    async def corrupt(payload: StoredPayload) -> bytes:
        return b"other"

    with pytest.raises(AIError) as raised:
        await load_stored_input_view(stored, corrupt)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
