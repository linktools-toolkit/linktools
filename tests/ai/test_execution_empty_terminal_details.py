#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empty diagnostic slots need no content owner; real diagnostics remain strict."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.execution import (ExecutionRecoveryDetail, RecoveryHandoffPhase,
                                  RecoveryTerminalHandoff, RecoveryTerminalOutcome)
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentRecord
from linktools.ai.persistence.codec import decode_envelope, payload_reference, runtime_policy_reference
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore

from .test_archive_closure import Offline
from .test_step_projection import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("details_kind", ("empty", "safe", "diagnostic"))
async def test_terminal_details_null_and_fixed_owner_restore(tmp_path: Path, details_kind: str) -> None:
    value = _execution()
    safe = {"reason": "retained"} if details_kind == "safe" else {}
    diagnostics = (ErrorDiagnostics("RuntimeError", "retained", "a" * 64)
                   if details_kind == "diagnostic" else None)
    outcome = RecoveryTerminalOutcome(ExecutionStatus.FAILED, ErrorCode.EXECUTION_FAILED.value, safe,
        StopReason.ERROR, None, None, UsageMetrics(), ExecutionEventType.EXECUTION_FAILED, {},
        value.updated_at, error_diagnostics=diagnostics)
    value = replace(value, status=ExecutionStatus.FAILED, error_code=ErrorCode.EXECUTION_FAILED.value,
        safe_error_details=safe, error_diagnostics=diagnostics,
        recovery=ExecutionRecoveryDetail(agent_run_id="run", handoff_phase=RecoveryHandoffPhase.PREPARED,
            terminal_handoff=RecoveryTerminalHandoff(outcome, "run", None)))
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    objects = InMemoryObjectStore()
    limits = SnapshotLimits(4096, 16 * 1024 * 1024)
    async with Persistence.open("details", storage=storage, tenant_id="tenant") as state:
        repository = state.execution.executions
        await repository.create_execution(value)
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        fields = decode_envelope(raw.data).value["payload"]["fields"]
        handoff = fields["recovery"]["fields"]["terminal_handoff"]["fields"]
        assert "safe_error_details" not in fields and "error_diagnostics" not in fields
        if details_kind == "empty":
            assert fields["terminal_details"] is None and handoff["terminal_details"] is None
        else:
            assert payload_reference(fields["terminal_details"]) == payload_reference(handoff["terminal_details"])
        assert await repository.get_fresh("execution", tenant_id="tenant") == value
        assert (await repository.get_metadata_record("execution", tenant_id="tenant")).safe_error_details == safe
        timeline = (await repository.get_timeline_many(("execution",), tenant_id="tenant"))["execution"]
        assert timeline.safe_error_details == safe and timeline.error_diagnostics == diagnostics
        retention = await repository.read_content(lambda _group: repository.state_store.read(
            lambda tx: repository.get_retention_in_transaction(tx, "execution", tenant_id="tenant")))
        assert retention.recovery_handoff
        await state.purge_unreferenced_objects((), exclusive=Offline())
    async with Persistence.open("details", storage=storage, tenant_id="tenant", read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open("details", storage=PersistenceStorage.from_root(target),
                                tenant_id="tenant", read_only=True) as state:
        assert await state.execution.executions.get_fresh("execution", tenant_id="tenant") == value


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing", "body"))
async def test_nonempty_terminal_details_reject_damaged_content(tmp_path: Path, damage: str) -> None:
    value = replace(_execution(), safe_error_details={"reason": "retained"})
    async with Persistence.open("details", storage=PersistenceStorage.sqlite(tmp_path / "damage.db"),
                                tenant_id="tenant") as state:
        repository = state.execution.executions
        await repository.create_execution(value)
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        payload = decode_envelope(raw.data).value["payload"]
        reference = payload_reference(payload["fields"]["terminal_details"])
        assert reference is not None and runtime_policy_reference(payload) is not None
        _scope, archive = repository.state_store.storage_group.opened_archive
        records = await archive.read(lambda tx: tx.scan_records())
        owner = next(record for record in records if record.kind == "archive_content"
                     and ContentRecord.from_record(record).identity == reference[2])
        if damage == "missing":
            await archive.mutate(lambda tx: tx.delete_records((owner.key,)))
        else:
            damaged = replace(owner, storage_revision=owner.storage_revision + 1, payload_bytes=b"invalid")
            assert await archive.mutate(lambda tx: tx.replace_record(damaged,
                expected_storage_revision=owner.storage_revision))
        with pytest.raises(AIError) as error:
            await repository.get_fresh("execution", tenant_id="tenant")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
