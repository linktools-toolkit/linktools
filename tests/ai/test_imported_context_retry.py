#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retry preserves imported pre-input context and its memory isolation."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CaptureInputRequest, ExecutionInputContext, Runtime, RuntimeStorage
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.runtime.state import RuntimeDomain

from .test_captured_execution_context import _NoToolsModels


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_size", [16, 80_000], ids=["inline", "object"])
async def test_retry_preserves_imported_history_and_metadata(tmp_path: Path, metadata_size: int) -> None:
    group = CapabilityGroup("retry-context")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    context = ExecutionInputContext.from_messages(
        (ModelRequest(parts=[UserPromptPart("earlier question")]),
         ModelResponse(parts=[TextPart("earlier answer")])),
        session_metadata={"note": "x" * metadata_size},
        replace_history_system_prompt=True,
    )
    storage = RuntimeStorage.filesystem(tmp_path)
    async with Runtime.open("retry-context", models=_NoToolsModels(), storage=storage, capabilities=(group,)) as runtime:
        principal = Principal("owner", runtime.tenant_id)
        source = await runtime.agents.get().start("superseded prompt", principal=principal, input_context=context)
        assert (await source.wait()).status is ExecutionStatus.SUCCEEDED
        record = await storage.execution.executions.get(source.execution_id, tenant_id=principal.tenant_id)
        assert record.input_context.payload.kind == ("inline" if metadata_size == 16 else "object")
        retry = await source.retry("replacement prompt")
        assert (await retry.wait()).status is ExecutionStatus.SUCCEEDED
        capture = await runtime.executions.capture_input(retry.execution_id, CaptureInputRequest(principal, "retry-context-capture"))
        captured = await runtime._input_captures.read_agent(capture, principal=principal)
        assert captured.input_context.digest == context.digest
        interactions = await retry.model_interactions(include_content=True)
        request = str(interactions.items[0].request)
        assert "earlier question" in request and "earlier answer" in request
        assert "replacement prompt" in request
        assert "superseded prompt" not in request


@pytest.mark.asyncio
async def test_retry_imported_context_matches_native_fork_history(tmp_path: Path) -> None:
    group = CapabilityGroup("retry-parity")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("retry-parity", models=_NoToolsModels(), storage=RuntimeStorage.filesystem(tmp_path), capabilities=(group,)) as runtime:
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
            retry = await source.retry("replacement question")
            assert (await retry.wait()).status is ExecutionStatus.SUCCEEDED
            interactions = await retry.model_interactions(include_content=True)
            request = str(interactions.items[0].request)
            assert "historical question" in request
            assert "replacement question" in request
            assert "superseded question" not in request


@pytest.mark.asyncio
async def test_retry_imported_memory_uses_fresh_isolated_scope(tmp_path: Path) -> None:
    group = CapabilityGroup("retry-memory")
    group.agent("default", model="default", allow_tools=("*",), allow_skills=(), allow_subagents=())
    storage = RuntimeStorage.filesystem(tmp_path)
    context = ExecutionInputContext.from_messages((), memory={
        "facts.txt": {"content": "accepted memory", "version": "m1:" + "1" * 64},
    })
    async with Runtime.open("retry-memory", models=_NoToolsModels(), storage=storage, capabilities=(group,)) as runtime:
        principal = Principal("owner", runtime.tenant_id)
        agent = runtime.agents.get()
        with pytest.raises(AIError) as raised:
            await agent.start("question", principal=principal, input_context=context, memory_scope="production")
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        source = await agent.start("question", principal=principal, input_context=context)
        assert (await source.wait()).status is ExecutionStatus.SUCCEEDED
        source_record = await storage.execution.executions.get(source.execution_id, tenant_id=principal.tenant_id)
        source_memory = RuntimeMemoryStore(storage.memory, object_store=storage.object_store(RuntimeDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id=source.execution_id, memory_scope=source_record.memory_scope)
        baseline = await source_memory.read("facts.txt", max_chars=1000)
        await source_memory.write("facts.txt", "source execution edit", expected_version=baseline.version)
        retry = await source.retry("replacement question")
        assert (await retry.wait()).status is ExecutionStatus.SUCCEEDED
        retry_record = await storage.execution.executions.get(retry.execution_id, tenant_id=principal.tenant_id)
        assert retry_record.memory_scope != source_record.memory_scope
        retry_memory = RuntimeMemoryStore(storage.memory, object_store=storage.object_store(RuntimeDomain.MEMORY),
            namespace=runtime.namespace, tenant_id=principal.tenant_id, execution_id=retry.execution_id, memory_scope=retry_record.memory_scope)
        retained = await retry_memory.read("facts.txt", max_chars=1000)
        assert retained.content == baseline.content
        assert retained.version == baseline.version
        await retry_memory.write("facts.txt", "retry execution edit", expected_version=retained.version)
        assert (await source_memory.read("facts.txt", max_chars=1000)).content == "source execution edit"
