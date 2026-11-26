#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish deferred receipts under the existing canonical turn authority."""

from datetime import datetime, timezone

from ..core import ApprovalStatus, ExternalCallStatus, ToolOperationStatus
from ..errors import AIError, ErrorCode
from ..execution import PendingDeferredCall, PendingToolContinuation
from ..persistence.contracts import (
    ApprovalRecord, ExternalCallRecord, deferred_resource_id,
)
from ..persistence.turns import CanonicalDeferredFrontier, SplitStepWaiting


def bind_deferred_publication(runner, turns, checkpoint) -> None:
    approvals = checkpoint.approvals
    external = checkpoint.external_calls

    async def publish(turn, checkpoint_id, tool_bridge, requests, *, split_source=None, **trace_opts) -> bool:
        owner = approvals.state_store
        if external.state_store is not owner:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        staged = None
        if split_source is not None and tool_bridge._split_approval_ref is not None:
            if (len(requests.approvals) != 1 or requests.calls
                    or any(requests.metadata.values()) or trace_opts):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            staged = await tool_bridge._consume_split_approval(requests.approvals[0])
        operations = ({staged.tool_call_id: staged} if staged is not None else
                      {operation.tool_call_id: operation for operation in await tool_bridge.list_operations()})

        def pending(call) -> PendingDeferredCall:
            operation = operations.get(call.tool_call_id)
            if (
                operation is None or operation.execution_id != turn.key.turn_id
                or operation.agent_run_id != turn.key.turn_id or operation.tool_name != call.tool_name
                or operation.status != ToolOperationStatus.PENDING
                or operation.producer_generation != turn.generation
                or operation.arguments_payload is None
            ):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return PendingDeferredCall(
                call.tool_call_id, call.tool_name, operation.arguments_payload,
                requests.metadata.get(call.tool_call_id, {}), capability_ref=operation.capability_ref,
                operation_id=operation.tool_operation_id,
            )

        identity = turn.key.turn_id
        continuation = PendingToolContinuation(
            source_agent_run_id=identity,
            approvals=tuple(map(pending, requests.approvals)),
            calls=tuple(map(pending, requests.calls)),
        )
        now = datetime.now(timezone.utc)

        async def receipts() -> None:
            for item in continuation.approvals:
                await approvals.create(ApprovalRecord(
                    deferred_resource_id("approval-v1", tool_bridge._tenant_id, identity, identity, item.tool_call_id),
                    identity, ApprovalStatus.PENDING, None, None, None, None, now, None,
                ))
            for item in continuation.calls:
                await external.create_call(ExternalCallRecord(
                    deferred_resource_id("external-call-v1", tool_bridge._tenant_id, identity, identity, item.tool_call_id),
                    identity, ExternalCallStatus.PENDING, None, now, None,
                ))

        if split_source is not None:
            leaves = (*continuation.approvals, *continuation.calls)
            if len(leaves) != 1 or trace_opts:
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            item = leaves[0]
            if item.operation_id is None:
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            approved = bool(continuation.approvals)
            framework_id = deferred_resource_id(
                "approval-v1" if approved else "external-call-v1",
                tool_bridge._tenant_id, identity, identity, item.tool_call_id,
            )
            frontier = CanonicalDeferredFrontier.from_json(turn.key, turn.generation, {
                "checkpoint_id": checkpoint_id,
                "approval_operation_ids": [item.operation_id] if approved else [],
                "external_operation_ids": [] if approved else [item.operation_id],
                "operation_metadata": {item.operation_id: dict(item.metadata)},
            })
            return await split_source.publish_waiting(
                frontier, waiting=SplitStepWaiting(
                    "approval" if approved else "external", item.operation_id, framework_id, turn.generation,
                ),
                checkpoint=checkpoint, tenant_id=tool_bridge._tenant_id, max_bytes=turn.terminal_bytes,
                publish_receipts=receipts, staged_operation=staged,
            )

        return await turns.publish_waiting_turn(
            turn.key, generation=turn.generation, checkpoint_id=checkpoint_id,
            continuation=continuation, receipt_store=owner, publish_receipts=receipts,
            max_bytes=turn.terminal_bytes, **trace_opts,
        )

    async def stop(frontier):
        tools = checkpoint.tools
        owner = approvals.state_store
        if external.state_store is not owner or tools.state_store is not owner:
            raise AIError(ErrorCode.STORAGE_ATOMIC_BATCH_UNSUPPORTED)
        turn_id = frontier.key.turn_id
        tenant_id = tools.tenant_id
        approval_ids = frontier.approval_operation_ids
        external_ids = frontier.external_operation_ids
        selected = set((*approval_ids, *external_ids))

        async def cancel_receipts(group) -> bool:
            records = await tools.list_by_agent_run(turn_id, tenant_id=tenant_id, execution_id=turn_id)
            by_id = {record.tool_operation_id: record for record in records}
            if len(by_id) != len(records) or any(
                (record := by_id.get(identity)) is None or record.execution_id != turn_id
                or record.agent_run_id != turn_id or record.producer_generation != frontier.generation
                for identity in selected
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            async def withdraw(repository, ids, *, prefix, stamp):
                identities = tuple(
                    deferred_resource_id(prefix, tenant_id, turn_id, turn_id, by_id[identity].tool_call_id)
                    for identity in ids
                )
                await repository.cancel_pending_in_transaction(
                    group.transaction(owner), identities, execution_id=turn_id, tenant_id=tenant_id, **stamp,
                )

            now = datetime.now(timezone.utc)
            await withdraw(approvals, approval_ids, prefix="approval-v1", stamp={"decided_at": now})
            await withdraw(external, external_ids, prefix="external-call-v1", stamp={"cancelled_at": now})
            await tools.cancel_parked_operations_in_transaction(
                group.transaction(owner), tuple(by_id[identity] for identity in (*approval_ids, *external_ids)),
                cancelled_at=now,
            )
            return all(
                record.status in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED, ToolOperationStatus.CANCELLED}
                or record.status is ToolOperationStatus.PENDING and record.tool_operation_id in selected
                for record in records
            )

        return await turns.request_parked_cancel(frontier, receipt_store=owner, cancel_receipts=cancel_receipts)

    runner.bind_deferred_publication(publish, stop)
