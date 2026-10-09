#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incremental history snapshots retain coherent pending-part streams."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_envelope,
    encode_envelope,
)
from linktools.ai.runtime.state._contracts import TranscriptChunk, TranscriptHeadRecord
from linktools.ai.runtime.state._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
)
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._store import stream_digest


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", (None, "future-index", "count", "duplicate-key", "body-domain"))
async def test_snapshot_pending_parts_are_bound_to_owner_and_current_head(
    tmp_path: Path, corruption: str | None,
) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    await storage.initialize(namespace="pending-snapshot", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        await archive.register_agent_run(AgentRunRecord("run"))
        repository = archive.transcript_repository
        message = ModelResponse(parts=[TextPart("first"), TextPart("second")])
        prepared = await repository.prepare_observation(
            "run", (), first_message_index=0, pending=message,
            pending_keys=("part:0", "part:1"),
        )
        await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = await archive.state_store.read(lambda tx: tx.scan_facts())
        aliases, sequences = canonical_snapshot_indexes(
            namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
            records=records, facts=facts, operations=(),
        )
        assert not sequences
        if corruption in {"future-index", "body-domain", "duplicate-key"}:
            changed = []
            for fact in facts:
                envelope = dict(decode_envelope(fact.data).value)
                chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
                if corruption == "future-index":
                    chunk = replace(chunk, first_message_index=1)
                    fact = replace(fact, stream_digest=stream_digest(
                        "pending-snapshot", "tenant", RuntimeDomain.RECOVERY.value,
                        "transcript_pending_parts", ["run", 1],
                    ))
                elif corruption == "body-domain":
                    chunk = replace(chunk, content=replace(chunk.content, source_domain=RuntimeDomain.EXECUTION))
                else:
                    envelope["pending_key"] = "part:0"
                envelope["payload"] = _encode_persisted_domain(chunk)
                changed.append(replace(fact, data=encode_envelope(envelope)))
            facts = tuple(changed)
        elif corruption == "count":
            changed = []
            for record in records:
                if record.kind == "transcript_head":
                    head = _decode_enveloped_domain(record.data, TranscriptHeadRecord)
                    record = replace(record, data=encode_envelope({
                        "type": "transcript_head",
                        "payload": _encode_persisted_domain(replace(head, pending_part_count=3)),
                    }))
                changed.append(record)
            records = tuple(changed)
        if corruption is not None:
            with pytest.raises(AIError) as raised:
                validate_snapshot_domain(
                    namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
                    records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
                )
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            validate_snapshot_domain(
                namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
                records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
            )
            completed = await repository.prepare_observation(
                "run", (message,), first_message_index=0, pending=None, pending_keys=(),
            )
            await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, completed))
            records = await archive.state_store.read(lambda tx: tx.scan_records())
            facts = await archive.state_store.read(lambda tx: tx.scan_facts())
            aliases, sequences = canonical_snapshot_indexes(
                namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
                records=records, facts=facts, operations=(),
            )
            validate_snapshot_domain(
                namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
                records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
            )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_snapshot_checkpoint_cannot_exceed_observed_transcript(tmp_path: Path) -> None:
    from linktools.ai.runtime.state._contracts import StoredAgentRunCheckpoint
    from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint

    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    await storage.initialize(namespace="pending-snapshot", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        await archive.register_agent_run(AgentRunRecord("run"))
        await archive.save_checkpoint(AgentRunCheckpoint(
            "run", 1, [ModelResponse(parts=[TextPart("complete")])],
            transcript_message_count_before=0,
        ))
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = await archive.state_store.read(lambda tx: tx.scan_facts())
        aliases, sequences = canonical_snapshot_indexes(
            namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
            records=records, facts=facts, operations=(),
        )
        validate_snapshot_domain(
            namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
            records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
        )
        corrupted = []
        for fact in facts:
            if fact.kind == "step_checkpoint":
                checkpoint = _decode_enveloped_domain(fact.data, StoredAgentRunCheckpoint)
                fact = replace(fact, data=encode_envelope({
                    "type": "stored_agent_run_checkpoint",
                    "payload": _encode_persisted_domain(replace(checkpoint, transcript_message_count=2)),
                }))
            corrupted.append(fact)
        with pytest.raises(AIError) as raised:
            validate_snapshot_domain(
                namespace="pending-snapshot", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
                records=records, aliases=aliases, facts=tuple(corrupted), operations=(), sequences=sequences,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()
