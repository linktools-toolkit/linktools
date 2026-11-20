#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retention reuses decoded owner metadata; source CAS remains the closing authority."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence.state import Observed, active_state_scope

from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("observed", "fallback", "stale"))
async def test_retention_observed_wire_preserves_policy_and_physical_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    async with Persistence.open("native-result", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(tmp_path / "retention.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "native result", "native result")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        observed = await repository.get_observed("execution", tenant_id="tenant")
        assert observed is not None and not observed.value.retention_closed
        projection = repository._execution_content.projection(observed.value)
        assert projection is not None and projection.storage_revision == observed.storage_revision
        if case == "fallback":
            # A caller-owned equal record has no source encoding to reuse.
            observed = Observed(replace(observed.value), observed.storage_revision)
        elif case == "stale":
            key = repository.execution_record_key("execution")
            async def advance(tx):
                stored = await tx.get_record(key)
                assert await tx.replace_record(replace(stored, storage_revision=stored.storage_revision + 1),
                                               expected_storage_revision=stored.storage_revision)
            await repository.state_store.mutate(advance)
        original_read = ExecutionContent.read_payloads
        original_policy = ContentArchive.capture_content_records
        original_encode = ExecutionContent.encode
        bodies, policies, encodings = [], [], []

        async def read_bodies(self, *args, **kwargs):
            bodies.append(True)
            assert case == "fallback", "unchanged observed wire restored Native bodies"
            return await original_read(self, *args, **kwargs)

        async def read_policy(self, identities, **kwargs):
            captured, load = await original_policy(self, identities, **kwargs)
            if any(content.codec == "runtime-policy" for content in captured):
                scope = active_state_scope()
                assert scope is not None and scope.writable
                assert self.store in scope.transactions
                policies.append(True)
            return captured, load

        async def encode(self, *args, **kwargs):
            encodings.append(True)
            assert case == "fallback", "observed encoding was needlessly recreated"
            return await original_encode(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(ExecutionContent, "read_payloads", read_bodies)
            patch.setattr(ContentArchive, "capture_content_records", read_policy)
            patch.setattr(ExecutionContent, "encode", encode)
            close = repository.state_store.mutate(lambda tx: repository.close_retention_in_transaction(
                tx, "execution", tenant_id="tenant", observed=observed))
            if case == "stale":
                with pytest.raises(AIError) as raised:
                    await close
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            else:
                closed, value = await close
                assert closed and value.retention_closed
        assert bool(policies) is (case == "fallback")
        assert bool(encodings) is (case == "fallback")
        assert not bodies
        restored = await repository.get_fresh("execution", tenant_id="tenant")
        assert restored.result == observed.value.result and restored.binding == observed.value.binding
        assert restored.retention_closed is (case != "stale")


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ("terminal", "hold", "handoff"))
async def test_observed_retention_preserves_closure_authority_without_content_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str,
) -> None:
    from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics
    from linktools.ai.execution import ExecutionRecoveryDetail, RecoveryHandoffPhase, RecoveryTerminalHandoff, RecoveryTerminalOutcome
    from .test_retention_read_io import _execution

    value = replace(_execution(), status=ExecutionStatus.FAILED,
                    dependency_hold_ids=("dependent",) if condition == "hold" else ())
    if condition == "handoff":
        outcome = RecoveryTerminalOutcome(ExecutionStatus.FAILED, ErrorCode.EXECUTION_FAILED.value,
            {}, StopReason.ERROR, None, None, UsageMetrics(), ExecutionEventType.EXECUTION_FAILED,
            {}, value.updated_at)
        value = replace(value, recovery=ExecutionRecoveryDetail(
            handoff_phase=RecoveryHandoffPhase.PREPARED,
            terminal_handoff=RecoveryTerminalHandoff(outcome, None, None)))
    async with Persistence.open("retention-owner", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(tmp_path / "owner.db")) as state:
        repository = state.execution.executions
        await repository.create_execution(value)
        observed = await repository.get_observed(value.execution_id, tenant_id="tenant")
        assert observed is not None

        async def unexpected_read(*args, **kwargs):
            pytest.fail("closing an observed owner must not reload immutable content")

        with monkeypatch.context() as patch:
            patch.setattr(ExecutionContent, "expand_policies", unexpected_read)
            patch.setattr(ExecutionContent, "expand_projection_components", unexpected_read)
            released, retained = await repository.state_store.mutate(
                lambda tx: repository.close_retention_in_transaction(
                    tx, value.execution_id, tenant_id="tenant", observed=observed))
        assert released is (condition == "terminal")
        assert retained.retention_closed is (condition == "terminal")
        assert retained.dependency_hold_ids == value.dependency_hold_ids
        assert retained.recovery_handoff is (condition == "handoff")
        fresh = await repository.get_fresh(value.execution_id, tenant_id="tenant")
        assert fresh.retention_closed is (condition == "terminal")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("present", "missing", "stale_missing"))
async def test_observed_retention_reads_clock_and_root_evidence_together(
    tmp_path: Path, case: str,
) -> None:
    from sqlalchemy import delete, event

    from linktools.ai.persistence._archive_closure import iter_native_message_roots
    from linktools.ai.steps import native_store_locator

    async with Persistence.open(
        "native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "clock.db"),
    ) as state:
        commands, commit, current, checkpoint = await _prepare(state, "native result", "native result")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        store = repository.state_store
        observed = await repository.get_observed("execution", tenant_id="tenant")
        assert observed is not None
        key = repository.execution_record_key("execution")
        before = await store.read(lambda tx: tx.get_record(key))
        assert before is not None
        native = tuple(iter_native_message_roots(before.data))
        assert native
        target = native[0]

        if case == "stale_missing":
            assert await store.mutate(lambda tx: tx.replace_record(
                replace(before, storage_revision=before.storage_revision + 1),
                expected_storage_revision=before.storage_revision,
            ))
            before = await store.read(lambda tx: tx.get_record(key))
        engine = store.storage_group.context.engine
        if case != "present":
            sequences = store.storage_group.metadata.tables["ai_step_sequences"]
            async with engine.begin() as connection:
                removed = await connection.execute(delete(sequences).where(
                    sequences.c.store_locator == native_store_locator(target.store_key),
                    sequences.c.store_key == target.store_key, sequences.c.sha256 == target.slices[0].first,
                ))
                assert removed.rowcount == 1

        reads = []

        def observe(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                reads.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", observe)
        try:
            closing = store.mutate(lambda tx: repository.close_retention_in_transaction(
                tx, "execution", tenant_id="tenant", observed=observed,
            ))
            if case == "present":
                closed, value = await closing
                assert closed and value.retention_closed
            else:
                with pytest.raises(AIError) as failure:
                    await closing
                assert failure.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", observe)

        combined = tuple(sql for sql in reads if "observed_at" in sql and "ai_step_sequences" in sql)
        rereads = tuple(sql for sql in reads if "observed_at" not in sql and "ai_step_sequences" in sql)
        assert len(combined) == 1
        if case == "present":
            assert reads == list(combined)
        else:
            assert bool(rereads) is (case == "missing")
            assert await store.read(lambda tx: tx.get_record(key)) == before
