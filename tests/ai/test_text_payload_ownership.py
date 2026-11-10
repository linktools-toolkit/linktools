#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text and structured values stay with their logical persistence owners."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.memory import PersistentMemoryStore
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.runtime._tool import RuntimeToolOperationBridge
from linktools.ai.storage import InMemoryObjectStore


@pytest.mark.asyncio
@pytest.mark.parametrize("codec", ("json", "model-message"))
async def test_large_archive_text_stays_in_state_even_with_object_storage(codec: str) -> None:
    objects = InMemoryObjectStore()
    store = InMemoryStateStore()
    await store.initialize()
    archive = ContentArchive(store, "scope", namespace="text", tenant_id="tenant", object_store=objects)
    text = "Long ordinary text 中文🙂\n" * 20000
    raw = (canonical_json_bytes({"content": text}) if codec == "json"
           else encode_model_messages((ModelRequest(parts=[ToolReturnPart("tool", text, tool_call_id="call")]),)))
    prepared = PreparedArchive()
    identities = await archive.prepare_content(((codec, raw),), prepared)
    await store.mutate(lambda transaction: archive.publish(transaction, prepared))
    assert await archive.read_contents(identities, codec=codec) == (raw,)
    assert [item async for item in objects.list_objects()] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_large_memory_text_is_owned_by_the_memory_record(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("memory-text", tenant_id="tenant", storage=storage) as state:
        objects = InMemoryObjectStore()
        memory = PersistentMemoryStore(
            state.memory, object_store=objects, namespace=state.namespace,
            tenant_id=state.tenant_id, execution_id="execution", memory_scope="notes",
        )
        text = "Memory remains ordinary text.\n" * 10000
        await memory.write("memory/note.md", text, expected_version=None)
        captured = await memory.capture()
        assert captured["memory/note.md"]["content"] == text
        assert [item async for item in objects.list_objects()] == []


def _tool_bridge(objects: InMemoryObjectStore) -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        None, objects, namespace="text", tenant_id="tenant", execution_id="execution",
        agent_run_id="run", binding_digest="a" * 64, owner="owner", background_tasks=set(),
    )


@pytest.mark.asyncio
async def test_tool_text_arguments_and_opaque_json_do_not_become_objects() -> None:
    objects = InMemoryObjectStore()
    bridge = _tool_bridge(objects)
    value = {"kind": "binary", "data": "not an attachment" * 20000}
    arguments = await bridge._arguments_payload(value)
    raw = encode_model_messages((ModelRequest(parts=[ToolReturnPart("tool", value, tool_call_id="call")]),))
    result = await bridge._payload(raw)
    assert arguments.kind == result.kind == "inline"
    assert await bridge._payload_bytes(result) == raw
    assert [item async for item in objects.list_objects()] == []


@pytest.mark.asyncio
async def test_tool_attachments_keep_surrounding_text_in_the_state_payload() -> None:
    objects = InMemoryObjectStore()
    bridge = _tool_bridge(objects)
    body = b"\x00\xffattachment"
    raw = encode_model_messages((ModelRequest(parts=[ToolReturnPart(
        "tool", ["Text stays in the owner", BinaryContent(body, media_type="application/octet-stream")],
        tool_call_id="call",
    )]),))
    result = await bridge._payload(raw)
    assert result.kind == "inline"
    assert result.encoding == "json-attachments"
    assert await bridge._payload_bytes(result) == raw
    files = [item async for item in objects.list_objects()]
    assert len(files) == 1 and files[0].size == len(body)
