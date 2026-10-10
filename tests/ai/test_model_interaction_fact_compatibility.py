#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable interaction history remains readable beside current records."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._contracts import ModelInteractionRecord
from linktools.ai.runtime.state._step_archive import _encode_step, _step_subject
from linktools.ai.runtime.state._store import FactQuery, StoredFact

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_mixed_interaction_facts_and_records_preserve_paging_and_count(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        now = datetime.now(timezone.utc)
        old = ModelInteractionRecord(
            run.agent_run_id, 1, 1, "agent", None, {}, None, None, None,
            "CANCELLED", None, 1, None, now, now,
        )
        store = state.execution.executions.state_store
        stream = archive._stream(run.agent_run_id, "interaction")
        fact = StoredFact(
            stream, 1, archive._agent_run_key(run.agent_run_id),
            "model_interaction", _step_subject(old), old.status, _encode_step(old),
        )

        async def seed(transaction):
            owner = await transaction.get_record(fact.owner_key_digest)
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_version=owner.storage_version)
            await transaction.insert_fact(fact)
            await transaction.reserve_sequence(archive._sequence(run.agent_run_id, "interaction"), 1)

        await store.mutate(seed)
        assert await archive.list_model_interactions(agent_run_id=run.agent_run_id) == [old]
        new = replace(old, step_index=2, model_request_seq=2)
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), interactions=(old, new),
            execution_id="execution", producer_generation=1,
        )
        assert await archive.model_interaction_count(agent_run_id=run.agent_run_id) == 2
        assert await archive.list_model_interactions(agent_run_id=run.agent_run_id, limit=1) == [old]
        assert await archive.list_model_interactions(
            agent_run_id=run.agent_run_id, after_model_request_seq=1, limit=1,
        ) == [new]
        assert await archive.list_model_interactions(
            agent_run_id=run.agent_run_id, after_model_request_seq=2, limit=1,
        ) == []
        assert await store.read(lambda tx: tx.list_facts(FactQuery(stream))) == (fact,)
        assert await store.read(lambda tx: tx.get_record(archive._interaction_key(run.agent_run_id, 1))) is None

        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), interactions=(replace(old, duration_ns=2),),
                execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        # A copied canonical record has the same logical identity as its old fact.
        await store.mutate(lambda tx: tx.insert_records((archive._stored_interaction(old),)))
        assert await archive.list_model_interactions(agent_run_id=run.agent_run_id) == [old, new]
        assert await archive.model_interaction_count(agent_run_id=run.agent_run_id) == 2
