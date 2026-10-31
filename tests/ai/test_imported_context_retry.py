#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Retry preserves imported pre-input context and its memory isolation."""
from ._runtime_test_helpers import model_features

from linktools.ai.capability import ExecutionCapability, TraceCapability
from pathlib import Path

import pytest
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import CaptureInputRequest
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._archive import ContentArchive

from .test_captured_execution_context import _NoToolsModels


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_size", [16, 80_000], ids=["inline", "object"])
@pytest.mark.parametrize("trace", (False, True), ids=("ordinary", "durable-trace"))
async def test_retry_preserves_imported_history_and_metadata(tmp_path: Path, metadata_size: int, trace: bool) -> None:
    group = RuntimeFeatures("retry-context")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    context = ExecutionInputContext.from_messages(
        (ModelRequest(parts=[UserPromptPart("earlier question")]),
         ModelResponse(parts=[TextPart("earlier answer")])),
        session_metadata={"note": "x" * metadata_size},
        replace_history_system_prompt=True,
    )
    storage = PersistenceStorage.filesystem(tmp_path)
    if trace:
        group.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('retry-context', storage=storage) as persistence_39, Runtime.attach(persistence=persistence_39, features=model_features(_NoToolsModels(), group)) as runtime:
        principal = Principal("owner", runtime.tenant_id)
        source = await runtime.agents.get().start("superseded prompt", principal=principal, input_context=context)
        assert ((await source.wait()).result).status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(source.execution_id, tenant_id=principal.tenant_id)
        assert record.input_context.payload.kind == "inline"
        manifest = record.input_context.payload.decode()
        assert manifest["kind"] == "execution-context"
        assert manifest["semantic_digest"] == context.digest
        assert len(str(manifest)) < 4096
        scope, archive_store = await runtime._state.execution.executions.state_store.storage_group.open_archive()
        archive = ContentArchive(
            archive_store, scope, namespace=runtime.namespace, tenant_id=principal.tenant_id,
            object_store=runtime._state.archive_object_store(PersistenceDomain.EXECUTION),
        )
        note, = await archive.content_records((manifest["session_metadata"]["note"]["content_id"],))
        assert note.payload.kind == ("inline" if metadata_size == 16 else "object")
        source_id = source.execution_id
    async with Persistence.open('retry-context', storage=storage) as persistence_57, Runtime.attach(persistence=persistence_57, features=model_features(_NoToolsModels(), group)) as runtime:
        source = await runtime.executions.get(source_id, principal=principal)
        with capture_run_messages() as messages:
            retry = await source.retry("replacement prompt")
            assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        capture = await runtime.executions.capture_input(retry.execution_id, CaptureInputRequest(principal, "retry-context-capture"))
        captured = await runtime._input_captures.read_agent(capture, principal=principal)
        assert captured.input_context.digest == context.digest
        requests = [str(messages)]
        if trace:
            interactions = await retry.model_interactions(include_content=True)
            requests.append(str(interactions.items[0].request))
        for request in requests:
            assert "earlier question" in request and "earlier answer" in request
            assert "replacement prompt" in request
            assert "superseded prompt" not in request


@pytest.mark.asyncio
@pytest.mark.parametrize("trace", (False, True), ids=("ordinary", "durable-trace"))
async def test_retry_imported_context_matches_native_fork_history(tmp_path: Path, trace: bool) -> None:
    group = RuntimeFeatures("retry-parity")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    if trace:
        group.capability(TraceCapability(), scope="runtime")
    group.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('retry-parity', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_83, Runtime.attach(persistence=persistence_83, features=model_features(_NoToolsModels(), group)) as runtime:
        principal = Principal("owner", runtime.tenant_id)
        agent = runtime.agents.get()
        first = await agent.start("historical question", principal=principal)
        await first.wait()
        fork = await first.fork("superseded question")
        await fork.wait()
        capture = await runtime.executions.capture_input(fork.execution_id, CaptureInputRequest(principal, "fork-context"))
        captured = await runtime._input_captures.read_agent(capture, principal=principal)
        imported = await agent.start(captured.prompt, principal=principal, input_context=captured.input_context)
        await imported.wait()
        for source in (fork, imported):
            with capture_run_messages() as messages:
                retry = await source.retry("replacement question")
                assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
            requests = [str(messages)]
            if trace:
                interactions = await retry.model_interactions(include_content=True)
                requests.append(str(interactions.items[0].request))
            for request in requests:
                assert "historical question" in request
                assert "replacement question" in request
                assert "superseded question" not in request


@pytest.mark.asyncio
async def test_retry_imported_memory_uses_fresh_isolated_scope(tmp_path: Path) -> None:
    from linktools.ai.capability import MemoryCapability

    group = RuntimeFeatures("retry-memory")
    group.capability(MemoryCapability(scope="production"))
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.filesystem(tmp_path)
    context = ExecutionInputContext.from_messages((), memory={
        "facts.txt": {"content": "accepted memory", "version": "m1:" + "1" * 64},
    })
    async with Persistence.open('retry-memory', storage=storage) as persistence_116, Runtime.attach(persistence=persistence_116, features=model_features(_NoToolsModels(), group)) as runtime:
        principal = Principal("owner", runtime.tenant_id)
        agent = runtime.agents.get()
        with pytest.raises(TypeError, match="memory_scope"):
            await agent.start("question", principal=principal, input_context=context, memory_scope="production")
        source = await agent.start("question", principal=principal, input_context=context)
        assert ((await source.wait()).result).status is ExecutionStatus.SUCCEEDED
        source_record = await runtime._state.execution.executions.get(source.execution_id, tenant_id=principal.tenant_id)
        source_memory = RuntimeMemoryStore(runtime._state.memory, object_store=runtime._state.object_store(PersistenceDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id=source.execution_id, memory_scope=source_record.memory_scope)
        baseline = await source_memory.read("facts.txt", max_chars=1000)
        await source_memory.write("facts.txt", "source execution edit", expected_version=baseline.version)
        retry = await source.retry("replacement question")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        retry_record = await runtime._state.execution.executions.get(retry.execution_id, tenant_id=principal.tenant_id)
        assert retry_record.memory_scope != source_record.memory_scope
        retry_memory = RuntimeMemoryStore(runtime._state.memory, object_store=runtime._state.object_store(PersistenceDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id=retry.execution_id, memory_scope=retry_record.memory_scope)
        retained = await retry_memory.read("facts.txt", max_chars=1000)
        assert retained.content == baseline.content
        assert retained.version == baseline.version
        await retry_memory.write("facts.txt", "retry execution edit", expected_version=retained.version)
        assert (await source_memory.read("facts.txt", max_chars=1000)).content == "source execution edit"
