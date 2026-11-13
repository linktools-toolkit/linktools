#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Start preflight does not duplicate admission-owned identity checks."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal, settle_during_lookup", ((False, False), (True, False), (True, True)))
async def test_start_identity_reads_once_unless_local_settlement_publishes_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal: bool, settle_during_lookup: bool,
) -> None:
    release, entered = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("start-identity-reads")
    features.agent("default", model="default", allow_tools=())
    if terminal:
        features.capability(ExecutionCapability(retain_history=True,
            commit=CommitPolicy.terminal(buffer=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))), scope="runtime")
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("start-identity-reads", storage=storage) as state:
        async with Runtime.attach(persistence=state,
            features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get().create_session("session")
            service = runtime._execution_service
            repository = service._state.idempotency
            read = repository.get
            reads = []

            async def record_read(*args, **kwargs):
                reads.append((args, kwargs))
                return await read(*args, **kwargs)

            monkeypatch.setattr(repository, "get", record_read)
            try:
                execution = await session.start("question", idempotency_key="same")
                # Terminal admission checks source again after acquiring its live claim.
                assert len(reads) == (2 if terminal else 1)
                await asyncio.wait_for(entered.wait(), 5)
                reads.clear()
                if settle_during_lookup:
                    pending = service._runtime_bridge.pending_turns
                    lookup = pending.lookup

                    async def settle_then_lookup(**kwargs):
                        turn = pending.peek_request(tenant_id=kwargs["tenant_id"],
                            scope=kwargs["scope"], idempotency_key=kwargs["idempotency_key"])
                        assert turn is not None
                        release.set()
                        await execution.wait(timeout_seconds=10)
                        await pending.settle(turn)
                        assert pending.peek_request(tenant_id=kwargs["tenant_id"],
                            scope=kwargs["scope"], idempotency_key=kwargs["idempotency_key"]) is None
                        return await lookup(**kwargs)

                    monkeypatch.setattr(pending, "lookup", settle_then_lookup)
                replay = await session.start("question", idempotency_key="same")
                assert replay.execution_id == execution.execution_id
                assert len(reads) == (2 if settle_during_lookup else 1)
            finally:
                release.set()
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            assert calls == 1
