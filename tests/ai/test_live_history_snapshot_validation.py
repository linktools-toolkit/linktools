#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incremental history snapshots retain coherent pending-part streams."""

from dataclasses import replace
from pathlib import Path

import pytest

from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.steps import native_store_key
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_envelope,
    encode_envelope,
)
from linktools.ai.persistence.contracts import TranscriptChunk, TranscriptHeadRecord
from linktools.ai.persistence._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
)
from linktools.ai.persistence.state import stream_key

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", (None, "future-index", "count", "duplicate-key", "content-scope", "content-count"))
async def test_snapshot_pending_parts_are_bound_to_owner_and_current_head(
    tmp_path: Path, corruption: str | None,
) -> None:
    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        repository = archive.transcript_repository
        message = ModelResponse(parts=[TextPart("first"), TextPart("second")])
        prepared = await repository.prepare_observation(
            "run", (), first_message_index=0, pending=message,
            pending_keys=("part:0", "part:1"),
        )
        await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, prepared))
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        records = await archive._project_run_records(records)
        facts = tuple(sorted(await archive.state_store.read(lambda tx: tx.scan_facts()),
                             key=lambda fact: (fact.stream_key, fact.sequence)))
        aliases, sequences = canonical_snapshot_indexes(
            namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
            records=records, facts=facts, operations=(),
        )
        assert not sequences
        if corruption in {"future-index", "content-scope", "content-count", "duplicate-key"}:
            changed = []
            for fact in facts:
                envelope = dict(decode_envelope(fact.data).value)
                chunk = _decode_enveloped_domain(fact.data, TranscriptChunk)
                if corruption == "future-index":
                    chunk = replace(chunk, first_message_index=1)
                    fact = replace(fact, stream_key=stream_key(
                        "archive-boundaries", "tenant", PersistenceDomain.TRACE.value,
                        "transcript_pending_parts", ("run", "1"),
                    ))
                elif corruption == "content-scope":
                    chunk = replace(chunk, content=replace(chunk.content, store_key=native_store_key("0", "tenant")))
                elif corruption == "content-count":
                    chunk = replace(chunk, content=replace(chunk.content, slices=()))
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
                    namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                    records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
                )
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            validate_snapshot_domain(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
            )
            completed = await repository.prepare_observation(
                "run", (message,), first_message_index=0, pending=None, pending_keys=(),
            )
            await archive.state_store.mutate(lambda tx: repository.commit_observation(tx, completed))
            records = await archive.state_store.read(lambda tx: tx.scan_records())
            records = await archive._project_run_records(records)
            facts = tuple(sorted(await archive.state_store.read(lambda tx: tx.scan_facts()),
                                 key=lambda fact: (fact.stream_key, fact.sequence)))
            aliases, sequences = canonical_snapshot_indexes(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, facts=facts, operations=(),
            )
            validate_snapshot_domain(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
            )


@pytest.mark.asyncio
async def test_snapshot_checkpoint_cannot_exceed_observed_transcript(tmp_path: Path) -> None:
    from linktools.ai.persistence.contracts import StoredAgentRunCheckpoint
    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint

    async with _archive(tmp_path, "sqlite") as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        await archive.save_checkpoint(AgentRunCheckpoint(
            "run", 1, [ModelResponse(parts=[TextPart("complete")])],
            transcript_message_count_before=0,
        ), execution_id="execution")
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        records = await archive._project_run_records(records)
        facts = tuple(sorted(await archive.state_store.read(lambda tx: tx.scan_facts()),
                             key=lambda fact: (fact.stream_key, fact.sequence)))
        aliases, sequences = canonical_snapshot_indexes(
            namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
            records=records, facts=facts, operations=(),
        )
        validate_snapshot_domain(
            namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
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
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, aliases=aliases, facts=tuple(corrupted), operations=(), sequences=sequences,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", (None, "owner", "sequence", "state", "identity", "conflict", "gap"))
async def test_snapshot_model_interaction_records_validate_identity_and_sequence(
    tmp_path: Path, corruption: str | None,
) -> None:
    from .test_model_interaction_lifecycle_paging import _interaction

    async with _archive(tmp_path) as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        records = await archive._project_run_records(records)
        first = _interaction("run", 1)
        second = _interaction("run", 3 if corruption == "gap" else 2)
        record = archive._stored_interaction(first)
        if corruption == "owner":
            record = replace(record, parent_key=b"x" * 32)
        elif corruption == "sequence":
            record = replace(record, sort_key=b"m:00000000000000000002")
        elif corruption == "state":
            record = replace(record, state="SUCCEEDED")
        elif corruption == "identity":
            record = replace(record, key=b"x" * 32)
        elif corruption == "conflict":
            second = replace(first, duration_ns=first.duration_ns + 1)
        records = (*records, record, archive._stored_interaction(second))

        def validate() -> None:
            aliases, sequences = canonical_snapshot_indexes(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, facts=(), operations=(),
            )
            validate_snapshot_domain(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, aliases=aliases, facts=(), operations=(), sequences=sequences,
            )

        if corruption is None:
            validate()
        else:
            with pytest.raises(AIError) as raised:
                validate()
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("corruption", (None, "missing-event", "gap", "ahead", "wrong-owner", "duplicate-role"))
async def test_snapshot_trace_requires_complete_unique_authoritative_event_prefix(
    tmp_path: Path, backend: str, corruption: str | None,
) -> None:
    from linktools.ai.persistence._step_contracts import StepEvent
    from linktools.ai.persistence.codec import _encode_step_envelope
    from linktools.ai.persistence.snapshots import (
        _StateSnapshotReader, _SnapshotCaptureBudget, _capture_snapshot_reader, SnapshotLimits,
    )

    async with _archive(tmp_path, backend) as (_state, archive, run):
        await archive.register_agent_run(run, execution_id="execution")
        for request in ("01", "1"):
            await archive.append_event(
                StepEvent("run", "MODEL_REQUEST_STARTED", 1,
                          metadata={"linktools.ai.model_request_seq": request}),
                execution_id="execution",
            )
        captured = await _capture_snapshot_reader(
            _StateSnapshotReader(archive.state_store, native_run_scope=("archive-boundaries", "tenant")),
            budget=_SnapshotCaptureBudget(SnapshotLimits(1000, 2**20)), page_size=1,
        )
        records, facts = captured.records, captured.facts
        records = await archive._project_run_records(records)
        assert not any(record.kind == "history_association" for record in records)
        assert len(facts) == 2
        aliases, sequences = canonical_snapshot_indexes(
            namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
            records=records, facts=facts, operations=(),
        )
        if corruption == "missing-event":
            facts = facts[:-1]
        elif corruption == "gap":
            facts = facts[1:]
        elif corruption == "ahead":
            sequences = {key: value + 1 for key, value in sequences.items()}
        elif corruption == "wrong-owner":
            facts = (replace(facts[0], owner_key=b"wrong"), facts[1])
        elif corruption == "duplicate-role":
            event = _decode_enveloped_domain(facts[1].data, StepEvent)
            event = replace(event, metadata={"linktools.ai.model_request_seq": "01"})
            facts = (facts[0], replace(facts[1], data=_encode_step_envelope(event)))

        def validate() -> None:
            validate_snapshot_domain(
                namespace="archive-boundaries", tenant_id="tenant", domain=PersistenceDomain.TRACE,
                records=records, aliases=aliases, facts=facts, operations=(), sequences=sequences,
            )

        if corruption is None:
            validate()
        else:
            with pytest.raises(AIError) as raised:
                validate()
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
