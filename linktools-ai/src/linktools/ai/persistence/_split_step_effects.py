#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact sealed-source reconciliation of the existing tool effect receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..core import ToolOperationStatus
from ..errors import AIError, ErrorCode
from .state import SqlStateTransaction

if TYPE_CHECKING:
    from ..storage import SqlStorageContext
    from ._checkpoint_repositories import ToolRepositoryImpl
    from ._split_step_source import SplitStepSourceObservation, SqlSplitStepSource
    from .state import StateStore


class SqlSplitStepEffects:
    """Inspect one canonical run; never create a second effect owner or receipt."""

    def __init__(
        self, *, source: SqlSplitStepSource, context: SqlStorageContext, source_store: StateStore,
        tools: ToolRepositoryImpl, tenant_id: str,
    ) -> None:
        if tools.state_store.storage_group is not source_store.storage_group:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        self._source, self._context, self._store = source, context, source_store
        self._tools, self._tenant_id = tools, tenant_id

    async def require_resolved(
        self, observation: SplitStepSourceObservation, *, claim_id: str,
        expected_calls: Mapping[str, str],
    ) -> None:
        """Expected operation IDs and call IDs come from the pinned checkpoint.

        Also inspect calls issued after that checkpoint: their durable intent
        may exist even when the producer died before the next model boundary.
        A replay-safe declaration does not resolve an ambiguous external effect.
        """
        if observation.phase != "sealed":
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

        async def settle(group) -> bool:
            source_tx = group.transaction(self._store)
            tool_tx = group.transaction(self._tools.state_store)
            if not isinstance(source_tx, SqlStateTransaction):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            await source_tx.run_sql(self._context, lambda db: self._source.guard_sealed(db, observation))

            records = await self._tools.list_by_execution(observation.turn_id, tenant_id=self._tenant_id)
            operations = {record.tool_operation_id: record for record in records}
            if len(operations) != len(records):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for identity, call_id in expected_calls.items():
                operation = operations.get(identity)
                if operation is None or operation.tool_call_id != call_id:
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

            blocked = False
            for record in records:
                generation = record.producer_generation
                if (record.execution_id != observation.turn_id or record.agent_run_id != observation.turn_id
                        or generation is None or generation > observation.generation):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                if record.status is ToolOperationStatus.CLAIMED:
                    if generation == observation.generation and record.owner != claim_id:
                        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
                    await self._tools.reconcile_turn_claim_in_transaction(
                        tool_tx, record.tool_operation_id, execution_id=observation.turn_id,
                        producer_generation=observation.generation, producer_claim_id=claim_id, turn_stopping=True,
                    )
                    blocked = True
                elif (record.status is ToolOperationStatus.EFFECT_UNKNOWN
                      or record.status is ToolOperationStatus.PENDING and record.effect_started_at is not None):
                    # A prior applied/failed receipt is reusable. A legacy-like
                    # pending state after start is not evidence of "not applied."
                    blocked = True
            return blocked

        members = tuple(dict.fromkeys((self._store, self._tools.state_store)))
        blocked = await self._store.storage_group.mutate(members, settle)
        if blocked:
            # Keep the unknown result durable; do not raise within its write.
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
