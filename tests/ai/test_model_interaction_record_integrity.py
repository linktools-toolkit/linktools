#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interaction high waters require their authoritative mutable records."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.contracts import ModelInteractionRecord
from linktools.ai.persistence._snapshot_validation import canonical_snapshot_indexes, validate_snapshot_domain
from linktools.ai.persistence._step_archive import _encode_step
from linktools.ai.persistence.state import StoredFact

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_interaction_fact_cannot_replace_missing_authoritative_record(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        now = datetime.now(timezone.utc)
        old = ModelInteractionRecord(
            run.agent_run_id, 1, 1, "agent", None, {}, None, None, None,
            "CANCELLED", None, 1, None, now, now,
        )
        store = archive.state_store
        stream = archive._stream(run.agent_run_id, "interaction")
        fact = StoredFact(
            stream, 1, archive._agent_run_key(run.agent_run_id),
            "model_interaction", bytes.fromhex(canonical_sha256({"agent_run_id": old.agent_run_id, "model_request_seq": 1})), old.status, _encode_step(old),
        )

        async def seed(transaction):
            owner = await transaction.get_record(fact.owner_key_digest)
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.insert_fact(fact)
            await transaction.reserve_sequence(archive._sequence(run.agent_run_id, "interaction"), 1)

        await store.mutate(seed)
        with pytest.raises(AIError) as missing:
            await archive.list_model_interactions(agent_run_id=run.agent_run_id)
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        new = replace(old, step_index=2, model_request_seq=2)
        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), interactions=(old, new),
                execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await archive.model_interaction_count(agent_run_id=run.agent_run_id) == 1
        assert await store.read(lambda tx: tx.get_record(archive._interaction_key(run.agent_run_id, 2))) is None

        records = await store.read(lambda tx: tx.scan_records())
        facts = await store.read(lambda tx: tx.scan_facts())
        with pytest.raises(AIError) as indexes:
            canonical_snapshot_indexes(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, facts=facts, operations=(),
            )
        assert indexes.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as snapshot:
            validate_snapshot_domain(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, facts=facts, operations=(), aliases=(),
                sequences={archive._sequence(run.agent_run_id, "interaction"): 1},
            )
        assert snapshot.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
