#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failed owners release buffered tails only after authoritative revocation."""

import asyncio
import json
from collections.abc import Hashable, Sequence
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain, ToolOperationStatus, WriteBufferEntry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.model.messages import decode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _MissingPayloadBuffer(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=4 * 1024 * 1024)
        self.pending: dict[tuple[str, str], bytes | None] = {}
        self.missing: set[tuple[str, str]] = set()
        self.fail_ack = False
        self.close_calls = 0

    async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
        accepted = await super().append(scope, id, key, content, wait=wait)
        if accepted:
            self.pending[scope, id] = content
        return accepted

    async def read_batch(self, scope: str, *, limit: int) -> tuple[WriteBufferEntry, ...]:
        entries = await super().read_batch(scope, limit=limit)
        return tuple(entry for entry in entries if (scope, entry[0]) not in self.missing)

    async def ack(self, scope: str, ids: Sequence[str]) -> None:
        if self.fail_ack:
            raise OSError("temporary cleanup failure")
        await super().ack(scope, ids)
        for id in ids:
            self.pending.pop((scope, id), None)

    async def close(self) -> None:
        self.close_calls += 1

    def lose(self, format: str) -> None:
        selected = next(key for key, content in self.pending.items()
                        if content is not None and json.loads(content).get("format") == format)
        self.missing.add(selected)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_format", ("execution-payload-v1", "trace-projection-v1"))
async def test_revoked_failed_owner_retries_exact_cleanup_without_replaying_effect(
    tmp_path: Path, payload_format: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _MissingPayloadBuffer()
    assert await buffer.append("other-owner", "entry", "key", b"{}", wait=False)
    policy = CommitPolicy.terminal(buffer=buffer)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=buffer)
    calls = {"model": 0, "tool": 0, "next": 0}
    next_turn = False
    result = "confirmed external result" * 100
    receipt = None

    async with Persistence.open("revoked-buffer", storage=storage) as state:
        async def effect(ctx: AgentContext[None]) -> str:
            calls["tool"] += 1
            execution = await state.execution.executions.get(ctx.execution_id, tenant_id=ctx.principal.tenant_id)
            tools = await state.recovery.tools.list_by_execution(ctx.execution_id, tenant_id=ctx.principal.tenant_id)
            assert execution.status is ExecutionStatus.STARTED
            assert len(tools) == 1 and tools[0].status is ToolOperationStatus.CLAIMED
            return result

        async def model(messages, info):
            nonlocal receipt
            if next_turn:
                calls["next"] += 1
                return ModelResponse(parts=[TextPart("next turn")])
            calls["model"] += 1
            if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[ToolCallPart("effect", {}, tool_call_id="effect-call")])
            tools = await state.recovery.tools.list_by_execution(execution.execution_id, tenant_id=state.tenant_id)
            assert len(tools) == 1 and tools[0].status is ToolOperationStatus.COMPLETED
            receipt = tools[0]
            decoded = decode_model_messages(receipt.result_payload.decode())
            assert decoded[0].parts[0].content == result
            buffer.lose(payload_format)
            return ModelResponse(parts=[TextPart("lost current turn")])

        def features() -> RuntimeFeatures:
            values = RuntimeFeatures("revoked-buffer")
            values.capability(ExecutionCapability(retain_history=True, commit=policy))
            values.capability(TraceCapability(commit=policy))
            values.tool(effect, effect_policy="non_replay_safe")
            values.agent("default", model="default", allow_tools=("effect",))
            return model_features(_Models(_UsageFunctionModel(model)), values)

        async with Runtime.attach(persistence=state, features=features(), metrics=None,
                                  auto_recover=False) as original:
            session = await original.agents.get().create_session("session")
            execution = await session.start("first turn", idempotency_key="first")
            with pytest.raises(AIError) as failure:
                await execution.wait(timeout_seconds=10)
            assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert receipt is not None
            pending = set(buffer.pending)
            assert len(pending) > 1
            for _ in range(2):
                with pytest.raises(AIError) as failure:
                    await original.close()
                assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                assert set(buffer.pending) == pending

            async with Runtime.attach(persistence=state, features=features(), metrics=None,
                                      auto_recover=False) as successor:
                principal = successor.default_principal
                repeated = await successor.agents.get().session("session").start(
                    "first turn", idempotency_key="first",
                )
                assert repeated.execution_id == execution.execution_id
                cancelled = await successor.executions.cancel(execution.execution_id,
                    CancelExecutionRequest(principal, "discard-lost-turn", force=True))
                assert cancelled.cancelled
                current = await state.execution.executions.get(execution.execution_id, tenant_id=state.tenant_id)
                assert current.status is ExecutionStatus.CANCELLED

                backend = original._execution_service.runtime_backend()
                archive = backend._run_stores[PersistenceDomain.TRACE]
                source_release = asyncio.Event()
                source_task = asyncio.create_task(source_release.wait())
                backend._track_checkpoint_task(source_task, "source-settlement", execution.execution_id)
                load_execution = backend.load_execution
                require_revoked = archive.require_revoked_execution_producer

                async def read_terminal(execution_id, *, tenant_id):
                    if execution_id == execution.execution_id:
                        return current
                    return await load_execution(execution_id, tenant_id=tenant_id)

                async def require_quiescent_source(**kwargs):
                    assert source_task.done(), "revocation cleanup raced an owned source task"
                    return await require_revoked(**kwargs)

                monkeypatch.setattr(backend, "load_execution", read_terminal)
                monkeypatch.setattr(archive, "require_revoked_execution_producer", require_quiescent_source)
                buffer.fail_ack = True
                closing = asyncio.create_task(original.close())
                try:
                    await asyncio.sleep(0)
                    assert not closing.done()
                    assert set(buffer.pending) == pending
                finally:
                    source_release.set()
                with pytest.raises(AIError) as failure:
                    await closing
                assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                assert set(buffer.pending) == pending
                buffer.fail_ack = False
                await original.close()
                assert buffer.pending == {("other-owner", "entry"): b"{}"}
                assert await buffer.read_batch("other-owner", limit=1) == (("entry", "key", b"{}"),)
                assert buffer.close_calls == 0

                kept = await state.recovery.tools.list_by_execution(execution.execution_id, tenant_id=state.tenant_id)
                assert len(kept) == 1 and kept[0].status is ToolOperationStatus.COMPLETED
                assert kept[0].tool_operation_id == receipt.tool_operation_id
                assert kept[0].result_payload == receipt.result_payload
                next_turn = True
                following = await successor.agents.get().session("session").start(
                    "next turn", idempotency_key="next",
                )
                terminal = (await following.wait(timeout_seconds=10)).result
                assert terminal.status is ExecutionStatus.SUCCEEDED
                assert terminal.output == {"text": "next turn"}
                assert calls == {"model": 2, "tool": 1, "next": 1}
        assert buffer.pending == {("other-owner", "entry"): b"{}"}
        assert buffer.close_calls == 0
