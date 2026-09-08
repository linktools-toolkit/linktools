#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attachment-specific extension of the existing ToolOperation terminal boundary."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import linktools.ai.runtime.state._commands as commands_runtime
import linktools.ai.runtime.state._repositories as repositories_runtime

from ..core import JsonValue, ToolOperationStatus
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._tool import RuntimeToolOperationBridge, ToolOperationRecord
from .state._attachments import AttachmentResult
from .state._durability import CommitObservation, DurableCommitState, run_durable_commit
from .state._repositories import ToolRepositoryImpl
from .state._store import StateGroupTransaction, StateTransaction

_installed = False
_original_complete = RuntimeToolOperationBridge.complete
_original_complete_payload = ToolRepositoryImpl.complete_payload
_original_complete_in_transaction = ToolRepositoryImpl.complete_in_transaction
_original_commit_tool_terminal = commands_runtime.RuntimeStateCommands.commit_tool_terminal


class _AttachmentToolReturn(dict[str, JsonValue]):
    """Private model-visible dict carrying one trusted attachment activation fact."""

    def __init__(self, value: dict[str, JsonValue], result: AttachmentResult) -> None:
        super().__init__(value)
        if not isinstance(result, AttachmentResult):
            raise TypeError("result must be AttachmentResult")
        self.attachment_result = result


def attachment_tool_return(
    value: dict[str, JsonValue],
    result: AttachmentResult,
) -> dict[str, JsonValue]:
    """Build the private successful read_attachment result consumed by ToolOperation."""
    return _AttachmentToolReturn(value, result)


async def _complete_tool(
    self: RuntimeToolOperationBridge,
    decision: Any,
    result: Any,
) -> bool:
    if not isinstance(result, _AttachmentToolReturn):
        return await _original_complete(self, decision, result)
    attachment_result = result.attachment_result
    payload = await self._result_payload(decision, dict(result))

    async def finish() -> ToolOperationRecord:
        if self._terminal_commands is not None:
            return await self._terminal_commands.commit_tool_terminal(
                decision.operation_id,
                tenant_id=self._tenant_id,
                owner=self._owner,
                fence=decision.fence,
                result_payload=payload,
                attachment_result=attachment_result,
            )
        return await self._repository.complete_payload(
            decision.operation_id,
            tenant_id=self._tenant_id,
            owner=self._owner,
            fence=decision.fence,
            result_payload=payload,
            attachment_result=attachment_result,
        )

    async def readback() -> CommitObservation[ToolOperationRecord]:
        observed = await self._repository.get_operation(
            decision.operation_id,
            tenant_id=self._tenant_id,
        )
        if observed is None:
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        if observed.status is ToolOperationStatus.COMPLETED:
            if observed.owner != decision.owner or observed.fence != decision.fence:
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                )
            if (
                observed.result_payload != payload
                or observed.attachment_result != attachment_result
            ):
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                )
            return CommitObservation(DurableCommitState.COMMITTED, value=observed)
        if (
            observed.status is ToolOperationStatus.CLAIMED
            and observed.owner == decision.owner
            and observed.fence == decision.fence
        ):
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_EFFECT_UNKNOWN),
            )
        return CommitObservation(
            DurableCommitState.NOT_COMMITTED,
            error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
        )

    committed = await run_durable_commit(
        finish,
        readback,
        background_tasks=self._background_tasks,
    )
    if committed.state is DurableCommitState.COMMITTED:
        return committed.cancelled
    if committed.state is DurableCommitState.NOT_COMMITTED:
        if committed.error is not None:
            raise committed.error
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if committed.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from committed.error
    if isinstance(committed.error, AIError) and committed.error.code in {
        ErrorCode.TOOL_OPERATION_CONFLICT,
        ErrorCode.TOOL_RESULT_CONFLICT,
        ErrorCode.TOOL_EFFECT_UNKNOWN,
    }:
        raise committed.error
    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from committed.error


async def _complete_attachment_in_transaction(
    repository: ToolRepositoryImpl,
    transaction: StateTransaction,
    tool_operation_id: str,
    *,
    tenant_id: str,
    owner: str,
    fence: int,
    result_payload: StoredPayload,
    attachment_result: AttachmentResult,
) -> ToolOperationRecord:
    if tenant_id != repository._tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    record = await transaction.get_record(repository._tool_key(tool_operation_id))
    if record is None:
        raise AIError(ErrorCode.STORAGE_NOT_FOUND)
    current = await repository._decode(record, ToolOperationRecord)
    now = await transaction.now()
    if current.status is ToolOperationStatus.COMPLETED:
        if (
            current.owner == owner
            and current.fence == fence
            and current.result_payload == result_payload
            and current.attachment_result == attachment_result
        ):
            return current
        raise AIError(ErrorCode.TOOL_RESULT_CONFLICT)
    if current.status in {
        ToolOperationStatus.FAILED,
        ToolOperationStatus.EFFECT_UNKNOWN,
        ToolOperationStatus.CANCELLED,
    }:
        raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT)
    repositories_runtime._require_live_tool_lease(
        current,
        owner=owner,
        fence=fence,
        now=now,
    )
    value = replace(
        current,
        status=ToolOperationStatus.COMPLETED,
        result_payload=result_payload,
        attachment_result=attachment_result,
        lease_expires_at=None,
        updated_at=now,
    )
    await repository._replace_tool_in_transaction(transaction, record, value)
    return value


async def _complete_payload(
    self: ToolRepositoryImpl,
    tool_operation_id: str,
    *,
    tenant_id: str,
    owner: str,
    fence: int,
    result_payload: StoredPayload,
    attachment_result: AttachmentResult | None = None,
) -> ToolOperationRecord:
    if attachment_result is None:
        return await _original_complete_payload(
            self,
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            result_payload=result_payload,
        )

    async def attempt() -> ToolOperationRecord:
        return await self._store.mutate(
            lambda transaction: _complete_attachment_in_transaction(
                self,
                transaction,
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                result_payload=result_payload,
                attachment_result=attachment_result,
            )
        )

    return await self._retry_storage_conflict(attempt)


async def _complete_in_transaction(
    self: ToolRepositoryImpl,
    transaction: StateTransaction,
    tool_operation_id: str,
    *,
    tenant_id: str,
    owner: str,
    fence: int,
    result_payload: StoredPayload,
    attachment_result: AttachmentResult | None = None,
) -> ToolOperationRecord:
    if attachment_result is None:
        return await _original_complete_in_transaction(
            self,
            transaction,
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            result_payload=result_payload,
        )
    return await _complete_attachment_in_transaction(
        self,
        transaction,
        tool_operation_id,
        tenant_id=tenant_id,
        owner=owner,
        fence=fence,
        result_payload=result_payload,
        attachment_result=attachment_result,
    )


async def _commit_tool_terminal(
    self: commands_runtime.RuntimeStateCommands,
    tool_operation_id: str,
    *,
    tenant_id: str,
    owner: str,
    fence: int,
    result_payload: StoredPayload | None = None,
    error_code: str | None = None,
    error_payload: StoredPayload | None = None,
    attachment_result: AttachmentResult | None = None,
) -> ToolOperationRecord:
    if attachment_result is None:
        return await _original_commit_tool_terminal(
            self,
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            result_payload=result_payload,
            error_code=error_code,
            error_payload=error_payload,
        )
    if result_payload is None or error_code is not None or error_payload is not None:
        raise ValueError("attachment tool result requires one successful result payload")
    tools = self._tools
    if tools is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    stores = [tools.state_store]

    async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
        return await tools.complete_in_transaction(
            group.transaction(tools.state_store),
            tool_operation_id,
            tenant_id=tenant_id,
            owner=owner,
            fence=fence,
            result_payload=result_payload,
            attachment_result=attachment_result,
        )

    async def readback() -> CommitObservation[ToolOperationRecord]:
        observed = await tools.get_operation(tool_operation_id, tenant_id=tenant_id)
        if observed is None:
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        if observed.status is ToolOperationStatus.COMPLETED:
            if (
                observed.owner != owner
                or observed.fence != fence
                or observed.result_payload != result_payload
                or observed.attachment_result != attachment_result
            ):
                return CommitObservation(
                    DurableCommitState.NOT_COMMITTED,
                    error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                )
            return CommitObservation(DurableCommitState.COMMITTED, value=observed)
        if (
            observed.status is ToolOperationStatus.CLAIMED
            and observed.owner == owner
            and observed.fence == fence
        ):
            return CommitObservation(DurableCommitState.NOT_COMMITTED)
        if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED,
                error=AIError(ErrorCode.TOOL_EFFECT_UNKNOWN),
            )
        return CommitObservation(
            DurableCommitState.NOT_COMMITTED,
            error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
        )

    committed = await run_durable_commit(
        lambda: stores[0].storage_group.mutate(stores, callback),
        readback,
        background_tasks=self._background_tasks,
    )
    if committed.state is DurableCommitState.COMMITTED:
        if committed.value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if committed.cancelled:
            raise asyncio.CancelledError
        return committed.value
    if committed.state is DurableCommitState.NOT_COMMITTED:
        if committed.error is not None:
            raise committed.error
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if committed.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from committed.error
    if isinstance(committed.error, AIError) and committed.error.code in {
        ErrorCode.TOOL_OPERATION_CONFLICT,
        ErrorCode.TOOL_RESULT_CONFLICT,
        ErrorCode.TOOL_EFFECT_UNKNOWN,
    }:
        raise committed.error
    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from committed.error


def install_attachment_tool_state() -> None:
    """Install the attachment-only ToolOperation terminal extension once."""
    global _installed
    if _installed:
        return
    RuntimeToolOperationBridge.complete = _complete_tool
    ToolRepositoryImpl.complete_payload = _complete_payload
    ToolRepositoryImpl.complete_in_transaction = _complete_in_transaction
    commands_runtime.RuntimeStateCommands.commit_tool_terminal = _commit_tool_terminal
    _installed = True


__all__ = ["attachment_tool_return", "install_attachment_tool_state"]
