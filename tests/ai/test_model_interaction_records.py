#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical model-request lifecycle ownership and replay contracts."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._codec import _decode_step_envelope, _encode_step_envelope
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
)
from linktools.ai.runtime.state._snapshot_validation import (
    canonical_snapshot_indexes,
    validate_snapshot_domain,
)
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._store import RecordQuery
from linktools.ai.storage import StoredPayload


def _running(sequence: int = 1) -> ModelInteractionRecord:
    return ModelInteractionRecord(
        agent_run_id="run",
        step_index=sequence,
        model_request_seq=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=None,
        request_envelope=None,
        response_context=None,
        status="RUNNING",
        error_code=None,
        duration_ns=None,
        usage=None,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=None,
    )


def _prepared(value: ModelInteractionRecord) -> ModelInteractionRecord:
    return replace(
        value,
        model={"route_id": "default", "model": "prepared-model"},
        request_context=ContextProjection(()),
        request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b"{}"), RuntimeDomain.RECOVERY,
        ),
    )


def _terminal(value: ModelInteractionRecord, status: str = "SUCCEEDED") -> ModelInteractionRecord:
    return replace(
        value,
        status=status,
        response_context=ContextProjection(()) if status == "SUCCEEDED" else None,
        error_code=ErrorCode.MODEL_API_ERROR.value if status == "FAILED" else None,
        duration_ns=23,
        finished_at=value.started_at,
    )


def _storage(backend: str, path: Path) -> RuntimeStorage:
    if backend == "memory":
        return RuntimeStorage.in_memory()
    if backend == "filesystem":
        return RuntimeStorage.filesystem(path / "state")
    return RuntimeStorage.sqlite(path / "state.db")


@pytest.mark.parametrize("phase", (
    "admitted", "prepared", "succeeded", "failed-before-prepare", "interrupted",
    "failed-partial", "cancelled-partial",
))
def test_current_interaction_wire_round_trips_lifecycle(phase: str) -> None:
    value = _running()
    if phase in {"prepared", "succeeded"}:
        value = _prepared(value)
    if phase == "succeeded":
        value = _terminal(value)
    if phase == "failed-before-prepare":
        value = _terminal(value, "FAILED")
    if phase == "interrupted":
        value = replace(_prepared(value), status="INTERRUPTED")
    if phase in {"failed-partial", "cancelled-partial"}:
        value = replace(
            _terminal(_prepared(value), "FAILED" if phase == "failed-partial" else "CANCELLED"),
            response_context=ContextProjection(()),
        )
    assert _decode_step_envelope(_encode_step_envelope(value)) == value


def test_metadata_can_be_prepared_until_request_content_is_published() -> None:
    admitted = _running()
    prepared = _prepared(admitted)
    metadata = replace(admitted, model=prepared.model)
    admitted.validate_successor(metadata)
    metadata.validate_successor(prepared)
    terminal = _terminal(prepared)
    metadata.validate_successor(terminal)
    for previous, invalid in (
        (metadata, replace(metadata, model_request_seq=2)),
        (metadata, replace(metadata, step_index=2)),
        (metadata, replace(metadata, purpose="compaction")),
        (prepared, replace(prepared, model={"model": "changed-after-publication"})),
        (terminal, metadata),
    ):
        with pytest.raises(AIError) as error:
            previous.validate_successor(invalid)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_interaction_identity_survives_preparation_and_completion(
    tmp_path: Path, backend: str,
) -> None:
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run")
        admitted = _running()
        prepared = _prepared(admitted)
        terminal = _terminal(prepared)
        for value in (admitted, admitted, prepared, prepared, terminal, terminal):
            await archive.sync_projection(run, events=(), checkpoints=(), interactions=(value,))
            assert await archive.list_model_interactions(agent_run_id="run") == [value]
            assert await archive.model_interaction_count(agent_run_id="run") == 1

        for invalid in (
            admitted,
            replace(terminal, duration_ns=24),
            replace(terminal, step_index=2),
        ):
            with pytest.raises(AIError) as raised:
                await archive.sync_projection(run, events=(), checkpoints=(), interactions=(invalid,))
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await archive.list_model_interactions(agent_run_id="run") == [terminal]

        if backend != "memory":
            records = await archive.state_store.read(
                lambda tx: tx.list_records(RecordQuery(kind="model_interaction"))
            )
            assert len(records) == 1
            assert records[0].storage_version == 2
            facts = await archive.state_store.read(lambda tx: tx.scan_facts())
            assert not any(fact.kind == "model_interaction" for fact in facts)
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_prepared_request_is_immutable_and_failed_preparation_is_visible(
    tmp_path: Path, backend: str,
) -> None:
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run")
        first = _prepared(_running())
        failed = _terminal(_running(2), "FAILED")
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(first, failed))
        invalid = replace(first, request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b'{"changed":true}'), RuntimeDomain.RECOVERY,
        ))
        with pytest.raises(AIError) as raised:
            await archive.sync_projection(run, events=(), checkpoints=(), interactions=(invalid,))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await archive.list_model_interactions(agent_run_id="run") == [first, failed]
        assert await archive.model_interaction_count(agent_run_id="run") == 2
        with pytest.raises(AIError):
            await archive.sync_projection(
                run, events=(), checkpoints=(),
                interactions=(_terminal(first), replace(failed, duration_ns=24)),
            )
        assert await archive.list_model_interactions(agent_run_id="run") == [first, failed]
        await archive.sync_projection(
            run, events=(), checkpoints=(), interactions=(_terminal(first),),
        )
        assert await archive.list_model_interactions(agent_run_id="run") == [_terminal(first), failed]
        assert await archive.model_interaction_count(agent_run_id="run") == 2
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_concurrent_completion_keeps_exactly_one_terminal_result(
    tmp_path: Path, backend: str,
) -> None:
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run")
        prepared = _prepared(_running())
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(prepared,))
        terminals = (_terminal(prepared), _terminal(prepared, "FAILED"))
        outcomes = await asyncio.gather(*(
            archive.sync_projection(run, events=(), checkpoints=(), interactions=(value,))
            for value in terminals
        ), return_exceptions=True)
        assert sum(value is None for value in outcomes) == 1
        assert sum(isinstance(value, AIError) for value in outcomes) == 1
        current = await archive.list_model_interactions(agent_run_id="run")
        assert len(current) == 1 and current[0] in terminals
        assert await archive.model_interaction_count(agent_run_id="run") == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_canonical_interaction_pages_and_snapshot_sequences_include_running(
    tmp_path: Path, backend: str,
) -> None:
    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        archive = storage.run_store.read_store(RuntimeDomain.RECOVERY)
        run = AgentRunRecord("run")
        values = tuple(_running(sequence) for sequence in range(1, 13))
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=values)
        assert await archive.list_model_interactions(
            agent_run_id="run", after_model_request_seq=8, limit=3,
        ) == list(values[8:11])
        assert await archive.model_interaction_count(agent_run_id="run") == 12
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = await archive.state_store.read(lambda tx: tx.scan_facts())
        operations = await archive.state_store.read(lambda tx: tx.scan_operations())
        aliases, sequences = canonical_snapshot_indexes(
            namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
            records=records, facts=facts, operations=operations,
        )
        validate_snapshot_domain(
            namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.RECOVERY,
            records=records, aliases=aliases, facts=facts, operations=operations, sequences=sequences,
        )
        await archive.release_agent_run("run")
        assert await archive.list_model_interactions(agent_run_id="run") == []
        assert await archive.model_interaction_count(agent_run_id="run") == 0
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_recovered_producer_interrupts_unknown_requests_and_fences_old_writes(
    tmp_path: Path, backend: str,
) -> None:
    from linktools.ai.core import ExecutionStatus
    from .test_history_projection_conformance import _record

    storage = _storage(backend, tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        repository = storage.execution.executions
        execution = replace(_record(ExecutionStatus.STARTED, 1), revision=1)
        await repository.create_with_history_head(execution)
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(tx, execution)
        )
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        run = AgentRunRecord("run")
        prepared = replace(
            _prepared(_running()),
            request_envelope=RuntimePayloadRef(
                StoredPayload.inline_bytes(b"{}"), RuntimeDomain.EXECUTION,
            ),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), interactions=(prepared,),
            execution_id="execution", producer_generation=1,
        )
        await repository.state_store.mutate(
            lambda tx: repository.admit_history_producer_in_transaction(
                tx, replace(execution, revision=2),
            )
        )
        with pytest.raises(AIError) as raised:
            await archive.interrupt_model_interactions(
                agent_run_id="run", execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        await archive.interrupt_model_interactions(
            agent_run_id="run", execution_id="execution", producer_generation=2,
        )
        head = await repository.get_history_head("execution", tenant_id="tenant")
        interrupted = replace(prepared, status="INTERRUPTED")
        assert await archive.list_model_interactions(agent_run_id="run") == [interrupted]
        assert interrupted.duration_ns is None and interrupted.finished_at is None
        await archive.interrupt_model_interactions(
            agent_run_id="run", execution_id="execution", producer_generation=2,
        )
        assert await repository.get_history_head("execution", tenant_id="tenant") == head
        with pytest.raises(AIError) as raised:
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), interactions=(_terminal(prepared),),
                execution_id="execution", producer_generation=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await archive.list_model_interactions(agent_run_id="run") == [interrupted]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_snapshot_event_associations_remain_derived_from_canonical_facts(tmp_path: Path) -> None:
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime.state._step_contracts import StepEvent
    from .test_history_projection_conformance import _record

    storage = _storage("sqlite", tmp_path)
    await storage.initialize(namespace="request-lifecycle", tenant_id="tenant")
    try:
        await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
        archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
        await archive.sync_projection(
            AgentRunRecord("run"),
            events=(StepEvent(
                "run", "MODEL_REQUEST_STARTED", 1,
                metadata={"linktools.ai.model_request_seq": "1"},
            ),),
            checkpoints=(),
            execution_id="execution",
        )
        records = await archive.state_store.read(lambda tx: tx.scan_records())
        facts = await archive.state_store.read(lambda tx: tx.scan_facts())
        operations = await archive.state_store.read(lambda tx: tx.scan_operations())
        aliases, sequences = canonical_snapshot_indexes(
            namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
            records=records, facts=facts, operations=operations,
        )
        validate_snapshot_domain(
            namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
            records=records, aliases=aliases, facts=facts, operations=operations, sequences=sequences,
        )
        omitted = next(record for record in records if record.kind == "history_association")
        with pytest.raises(AIError) as missing:
            validate_snapshot_domain(
                namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
                records=tuple(record for record in records if record.key_digest != omitted.key_digest),
                aliases=aliases, facts=facts, operations=operations, sequences=sequences,
            )
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

        assert sum(record.kind == "history_association" for record in records) == 2
        corrupted = tuple(
            replace(record, data={"sequence": 2}) if record.kind == "history_association" else record
            for record in records
        )
        with pytest.raises(AIError) as raised:
            validate_snapshot_domain(
                namespace="request-lifecycle", tenant_id="tenant", domain=RuntimeDomain.EXECUTION,
                records=corrupted, aliases=aliases, facts=facts, operations=operations, sequences=sequences,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()
