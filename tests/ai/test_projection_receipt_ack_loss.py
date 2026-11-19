#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""An actual committed Trace mutation can lose its acknowledgment safely."""
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CreateSessionRequest
from linktools.ai.persistence import runs
from linktools.ai.persistence._sql import SqlStateStorageGroup, _SqlTransaction
from linktools.ai.persistence.durability import DurableCommitState
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _open_local_runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
async def test_projection_commit_ack_loss_reads_back_trace_without_delivering_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    notifications = []
    resolved = []
    lost = []
    failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    import inspect
    raw_records = []
    original_get = _SqlTransaction.get_record
    async def get_record(self, key):
        caller = inspect.currentframe().f_back.f_code.co_qualname
        record = await original_get(self, key)
        if caller.endswith("_commit_captured_execution_projection.<locals>.operation.<locals>.commit"):
            raw_records.append((self, record))
        return record
    monkeypatch.setattr(_SqlTransaction, "get_record", get_record)

    original_mutate = SqlStateStorageGroup.mutate
    async def mutate(self, stores, callback):
        result = await original_mutate(self, stores, callback)
        if (not lost and len(raw_records) >= 2
                and callback.__qualname__.endswith(
                    "_commit_captured_execution_projection.<locals>.operation.<locals>.commit")):
            (_, before), (transaction, after) = raw_records[-2:]
            assert before is not None and after is not None
            from dataclasses import replace
            assert after.storage_revision > before.storage_revision
            assert replace(after, storage_revision=before.storage_revision) == before
            store = next(store for store in stores
                         if store.store_key == transaction._store_key)
            receipt = store, before, after
            # The original mutation has exited its real SQL transaction. A new read
            # proves the owner revision was committed before acknowledgment is lost.
            assert await store.read_record(after.key) == after
            lost.append((receipt, result))
            raise failure
        return result
    monkeypatch.setattr(SqlStateStorageGroup, "mutate", mutate)

    original_commit = runs.run_durable_commit
    async def commit(operation, readback, **kwargs):
        result = await original_commit(operation, readback, **kwargs)
        if result.error is failure:
            resolved.append(result)
        return result
    monkeypatch.setattr(runs, "run_durable_commit", commit)

    original_bind = runs.RuntimeAgentRunStore.bind_step_boundary
    def bind(self, run_id, writer, require_active, *, confirm_projection=None):
        def confirm(*receipt):
            notifications.append(receipt)
            confirm_projection(*receipt)
        return original_bind(self, run_id, writer, require_active,
                             confirm_projection=None if confirm_projection is None else confirm)
    monkeypatch.setattr(runs.RuntimeAgentRunStore, "bind_step_boundary", bind)

    def respond(_messages, _info):
        return ModelResponse(parts=[TextPart("complete")])
    async with _open_local_runtime(Workspace.initialize(tmp_path), model=_Models(_UsageFunctionModel(respond))) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        run = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert (await run.wait(timeout_seconds=20)).result.output == {"text": "complete"}
    assert len(lost) == len(resolved) == 1
    receipt, committed_head = lost[0]
    assert resolved[0].state is DurableCommitState.COMMITTED
    assert resolved[0].value == committed_head
    assert committed_head.interaction_count > 0 or committed_head.event_count > 0
    assert all(delivered != receipt for delivered in notifications)
