#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct canonical deferred input from its accepted tool receipts."""

from pydantic_ai.tools import DeferredToolResults, ToolApproved, ToolDenied

from ..capability import PydanticToolControlCapability
from ..core import ApprovalStatus, ExternalCallStatus
from ..errors import AIError, ErrorCode
from ..execution import PendingDeferredCall, PendingToolContinuation
from ..persistence.contracts import deferred_resource_id
from ._session_deferred import SessionDeferredCommands
from ._tool import settle_canonical_deferred_leaf


def canonical_deferred_commands(
    turns, checkpoint, authorization, *, namespace, tenant_id, objects, read_payload, resume,
    split_source=None, turn_store=None,
):
    approvals = checkpoint.approvals
    calls = checkpoint.external_calls

    async def load(frontier):
        operation_ids = frontier.approval_operation_ids + frontier.external_operation_ids
        records = {
            record.tool_operation_id: record
            for record in await checkpoint.tools.get_operations(operation_ids, tenant_id=tenant_id)
        }
        turn_id = frontier.key.turn_id

        def pending(operation_id):
            operation = records.get(operation_id)
            if (operation is None or operation.execution_id != turn_id or operation.agent_run_id != turn_id
                    or operation.arguments_payload is None or operation.producer_generation != frontier.generation):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return PendingDeferredCall(
                operation.tool_call_id, operation.tool_name, operation.arguments_payload,
                frontier.operation_metadata.get(operation_id, {}), capability_ref=operation.capability_ref,
                operation_id=operation_id,
            )

        return PendingToolContinuation(
            turn_id, tuple(map(pending, frontier.approval_operation_ids)),
            tuple(map(pending, frontier.external_operation_ids)),
        )

    async def resolutions(frontier, continuation):
        turn_id = frontier.key.turn_id
        decisions, supplies = [], []
        for item in continuation.approvals:
            identity = deferred_resource_id("approval-v1", tenant_id, turn_id, turn_id, item.tool_call_id)
            record = await approvals.get(identity, tenant_id=tenant_id)
            if record is None or record.execution_id != turn_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is ApprovalStatus.PENDING:
                return None
            if record.status not in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED}:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            decisions.append((item, record))
        for item in continuation.calls:
            identity = deferred_resource_id("external-call-v1", tenant_id, turn_id, turn_id, item.tool_call_id)
            record = await calls.get(identity, tenant_id=tenant_id)
            if record is None or record.execution_id != turn_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is ExternalCallStatus.PENDING:
                return None
            if record.status is not ExternalCallStatus.SUPPLIED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            supplies.append((item, record))
        return decisions, supplies

    async def is_ready(frontier):
        loaded = await resolutions(frontier, await load(frontier))
        if loaded is None:
            return False
        decisions, supplies = loaded
        settled = [(item, record, None) for item, record in decisions if record.status is ApprovalStatus.DENIED]
        for item, record in supplies:
            if record.result_payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            settled.append((item, record, await read_payload(record.result_payload)))
        for pending, record, value in settled:
            await settle_canonical_deferred_leaf(
                checkpoint.tools, checkpoint_objects=objects, namespace=namespace, tenant_id=tenant_id,
                turn_id=frontier.key.turn_id, generation=frontier.generation, pending=pending,
                resolution=record, external_value=value,
            )
        return True

    async def resume_settled(frontier):
        continuation = await load(frontier)
        resolved = await resolutions(frontier, continuation)
        if resolved is None:
            return
        decisions, supplies = resolved
        results = DeferredToolResults()
        for pending, record in decisions:
            if record.status is ApprovalStatus.APPROVED:
                arguments = await read_payload(pending.arguments_payload)
                if not isinstance(arguments, dict):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                results.approvals[pending.tool_call_id] = ToolApproved(override_args=arguments)
            else:
                results.approvals[pending.tool_call_id] = ToolDenied(
                    message=record.decision_message or "The tool call was denied.",
                )
            results.metadata[pending.tool_call_id] = dict(record.resolution_metadata)
        for pending, record in supplies:
            if record.result_payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await read_payload(record.result_payload)
            if record.resolution_kind == "succeeded":
                results.calls[pending.tool_call_id] = result
            elif record.resolution_kind == "retry" and isinstance(result, str):
                results.calls[pending.tool_call_id] = PydanticToolControlCapability.build_model_retry(result)
            elif record.resolution_kind == "failed" and isinstance(result, str):
                results.calls[pending.tool_call_id] = PydanticToolControlCapability.build_tool_failed(result)
            else:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            results.metadata[pending.tool_call_id] = dict(record.resolution_metadata)
        await resume(frontier, continuation, results)

    return SessionDeferredCommands(
        turns, authorization, tenant_id=tenant_id, load=load, resume=resume_settled, is_ready=is_ready,
        split=None if split_source is None else (split_source, turn_store, checkpoint.tools.state_store),
    )
