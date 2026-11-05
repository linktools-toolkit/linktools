#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool mutations converge after retryable storage conflicts."""
from types import SimpleNamespace

import pytest
from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import StoredPayload
from .test_tool_cas_convergence import _admission


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("admit", "complete_payload", "fail_payload", "mark_effect_unknown"))
async def test_tool_mutation_retries_conflict_and_persists_result(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    async with Persistence.open("tool-retry", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        repository = state.checkpoint.tools
        if method != "admit":
            await repository.admit(_admission())
        mutate = repository.state_store.mutate
        injected = False

        async def conflict_once(operation):
            nonlocal injected
            if not injected:
                injected = True
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return await mutate(operation)

        monkeypatch.setattr(repository.state_store, "mutate", conflict_once)
        if method == "admit":
            result = await repository.admit(_admission())
        else:
            kwargs = {"result_payload": StoredPayload.inline_bytes(b"result")} if method == "complete_payload" else {
                "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value if method == "mark_effect_unknown" else ErrorCode.TOOL_EXECUTION_FAILED.value}
            if method == "fail_payload":
                kwargs["error_payload"] = StoredPayload.inline_bytes(b'{"version":1,"kind":"tool_call_failed","message":"failed"}')
            result = await getattr(repository, method)("tool-operation", tenant_id="tenant", owner="tool-owner", fence=1, **kwargs)
        assert injected
        assert result.status is {"admit": ToolOperationStatus.CLAIMED, "complete_payload": ToolOperationStatus.COMPLETED,
            "fail_payload": ToolOperationStatus.FAILED, "mark_effect_unknown": ToolOperationStatus.EFFECT_UNKNOWN}[method]
        assert await repository.get_operation("tool-operation", tenant_id="tenant") == result


@pytest.mark.asyncio
async def test_tool_reserve_rejects_malformed_failure_payload() -> None:
    async with Persistence.open("tool-retry", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        record = SimpleNamespace(status=ToolOperationStatus.FAILED, error_code=ErrorCode.TOOL_EXECUTION_FAILED.value,
                                 error_payload=StoredPayload.inline_bytes(b'{"version":1}'))
        with pytest.raises(AIError) as raised:
            await state.checkpoint.tools.reserve(record)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_tool_mutation_preserves_semantic_conflict_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("tool-retry", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        conflict = AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
        rejected = False

        async def reject(operation):
            nonlocal rejected
            assert not rejected, "semantic conflicts must not be retried"
            rejected = True
            raise conflict

        monkeypatch.setattr(state.checkpoint.tools.state_store, "mutate", reject)
        with pytest.raises(AIError) as raised:
            await state.checkpoint.tools.admit(_admission())
        assert raised.value is conflict
        assert rejected
