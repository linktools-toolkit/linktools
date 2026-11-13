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
