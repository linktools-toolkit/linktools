#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint publication batches reads without weakening its producer fence."""

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import update

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._sql import _SqlGroupTransaction
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from .test_checkpoint_write_batches import _snapshot
from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _producer(tmp_path, backend, *, checkpointed=True):
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="checkpoint-publication", tenant_id="tenant")
    try:
        repository = state.execution.executions
        execution = replace(_execution(), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        store = state.run_store
        store.bind_execution_producer(
            "run", execution_id="execution", producer_generation=1,
            conversation=False, restart_checkpoints=checkpointed, durable_trace=False,
        )
        await store.register_agent_run(AgentRunRecord("run"), execution_id="execution")
        yield state, store, repository, execution
    finally:
        await state.close()


def _checkpoint(count):
    return AgentRunCheckpoint("run", count, [
        ModelRequest(parts=[UserPromptPart(str(index), timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)) for index in range(count)
    ], transcript_message_count_before=0)


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("scalar", (False, True))
async def test_publication_preserves_scalar_group_fallback_and_replay(tmp_path, monkeypatch, backend, scalar):
    async with _producer(tmp_path, backend) as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        storage_group = repository.state_store.storage_group
        original = storage_group.mutate
        reads = []

        class ScalarGroup:
            def __init__(self, group):
                self.group = group

            def transaction(self, member):
                return self.group.transaction(member)

        class ReadingGroup(ScalarGroup):
            async def get_records(self, requests):
                reads.append(tuple(requests))
                return await self.group.get_records(requests)

        async def mutate(stores, callback):
            return await original(stores, lambda group: callback(
                ScalarGroup(group) if scalar else ReadingGroup(group)
            ))

        with monkeypatch.context() as patch:
            patch.setattr(storage_group, "mutate", mutate)
            checkpoint = _checkpoint(1)
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
            before = await _snapshot(archive.state_store)
            await store.save_checkpoint(checkpoint, execution_id="execution", producer_generation=1)
        assert await _snapshot(archive.state_store) == before
        assert await archive.checkpoint_count(agent_run_id="run") == 1
        assert (await archive.latest_checkpoint(agent_run_id="run")).messages == checkpoint.messages
        assert bool(reads) is (not scalar and backend == "sqlite")


@pytest.mark.parametrize("condition", ("cancel", "takeover", "no_checkpoint"))
async def test_publication_prefetch_follows_producer_fence(tmp_path, monkeypatch, condition):
    async with _producer(tmp_path, "sqlite", checkpointed=condition != "no_checkpoint") as (state, store, repository, execution):
        reads = []
        original = _SqlGroupTransaction.get_records

        async def read(group, requests):
            reads.append(requests)
            return await original(group, requests)

        if condition == "cancel":
            await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=1,
                next_record=replace(execution, revision=2, status=ExecutionStatus.CANCELLING))
        elif condition == "takeover":
            current = await repository.compare_and_swap("execution", tenant_id="tenant", expected_revision=1,
                next_record=replace(execution, revision=2))
            await repository.state_store.mutate(lambda tx: repository.admit_history_producer_in_transaction(tx, current))
        with monkeypatch.context() as patch:
            patch.setattr(_SqlGroupTransaction, "get_records", read)
            if condition == "no_checkpoint":
                await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
            else:
                with pytest.raises(AIError) as raised:
                    await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
                assert raised.value.code is (ErrorCode.EXECUTION_CANCELLED if condition == "cancel" else ErrorCode.STORAGE_CONFLICT)
        assert reads == []


@pytest.mark.parametrize("target", ("agent_run", "transcript_head", "context_projection", "failure"))
async def test_prefetched_owner_conflict_rolls_back_complete_checkpoint(tmp_path, monkeypatch, target):
    async with _producer(tmp_path, "sqlite") as (state, store, repository, _execution):
        archive = store.read_store(PersistenceDomain.CHECKPOINT)
        await store.save_checkpoint(_checkpoint(1), execution_id="execution", producer_generation=1)
        content = await archive.transcript_repository.content_archive()
        members = (repository.state_store, archive.state_store, content.store)
        before = [await _snapshot(member) for member in members]
        original = _SqlGroupTransaction.get_records
        injections = []

        async def read(group, requests):
            values = await original(group, requests)
            if archive.state_store in requests and not injections:
                injections.append(target)
                if target == "failure":
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                record = next(record for record in values[archive.state_store].values() if record.kind == target)
                tx = group.transaction(archive.state_store)
                table = tx._table("ai_state_records")
                await tx._execute(update(table).where(
                    table.c.store_digest == tx._store_hex,
                    table.c.key_digest == record.key_digest.hex(),
                ).values(storage_revision=record.storage_revision + 1))
            return values

        with monkeypatch.context() as patch:
            patch.setattr(_SqlGroupTransaction, "get_records", read)
            with pytest.raises(AIError) as raised:
                await store.save_checkpoint(_checkpoint(2), execution_id="execution", producer_generation=1)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert injections == [target]
        assert [await _snapshot(member) for member in members] == before
