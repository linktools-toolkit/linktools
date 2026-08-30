#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolOperation command convergence at the outer durable transaction boundary."""

import asyncio

from ...core import ToolOperationStatus
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._tool import ToolOperationRecord
from ._commands import RuntimeStateCommands as _RuntimeStateCommands
from ._contracts import ToolOperationAdmission
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._repositories import _tool_admission_matches
from ._store import StateGroupTransaction


class RuntimeStateCommands(_RuntimeStateCommands):
    """Add fresh-transaction convergence for ToolOperation durable commands."""

    async def commit_tool_admission(
        self,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        while True:
            try:
                return await super().commit_tool_admission(request)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                tools = self._tools
                if tools is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
                observed = await tools.get_operation(
                    request.tool_operation_id,
                    tenant_id=request.tenant_id,
                )
                if observed is None:
                    await asyncio.sleep(0)
                    continue
                if not _tool_admission_matches(observed, request):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                if observed.status is ToolOperationStatus.EFFECT_UNKNOWN:
                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
                if observed.status is ToolOperationStatus.CANCELLED:
                    raise AIError(ErrorCode.TOOL_OPERATION_CONFLICT) from error
                if observed.status in {
                    ToolOperationStatus.COMPLETED,
                    ToolOperationStatus.FAILED,
                }:
                    return observed
                # CLAIMED/PENDING must be classified by a fresh repository
                # transaction so lease expiry and owner takeover semantics are
                # never guessed from an out-of-transaction readback.
                await asyncio.sleep(0)

    async def commit_tool_terminal(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload | None = None,
        error_code: str | None = None,
        error_payload: StoredPayload | None = None,
    ) -> ToolOperationRecord:
        tools = self._tools
        if tools is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if result_payload is None and error_code is None:
            raise ValueError("tool terminal command requires a result or error")
        expected_status = (
            ToolOperationStatus.COMPLETED
            if result_payload is not None
            else ToolOperationStatus.FAILED
        )
        terminal_error_code = error_code or ErrorCode.EXECUTION_FAILED.value
        stores = [tools.state_store]
        cancelled = False

        async def callback(group: StateGroupTransaction) -> ToolOperationRecord:
            transaction = group.transaction(tools.state_store)
            if result_payload is not None:
                return await tools.complete_in_transaction(
                    transaction,
                    tool_operation_id,
                    tenant_id=tenant_id,
                    owner=owner,
                    fence=fence,
                    result_payload=result_payload,
                )
            return await tools.fail_in_transaction(
                transaction,
                tool_operation_id,
                tenant_id=tenant_id,
                owner=owner,
                fence=fence,
                error_code=terminal_error_code,
                error_payload=error_payload,
            )

        async def readback() -> CommitObservation[ToolOperationRecord]:
            observed = await tools.get_operation(
                tool_operation_id,
                tenant_id=tenant_id,
            )
            if observed is None:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if observed.status is expected_status:
                if observed.owner != owner or observed.fence != fence:
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                if result_payload is not None:
                    if observed.result_payload != result_payload:
                        return CommitObservation(
                            DurableCommitState.NOT_COMMITTED,
                            error=AIError(ErrorCode.TOOL_RESULT_CONFLICT),
                        )
                elif (
                    observed.error_code != terminal_error_code
                    or observed.error_payload != error_payload
                ):
                    return CommitObservation(
                        DurableCommitState.NOT_COMMITTED,
                        error=AIError(ErrorCode.TOOL_OPERATION_CONFLICT),
                    )
                return CommitObservation(
                    DurableCommitState.COMMITTED,
                    value=observed,
                )
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

        while True:
            result = await run_durable_commit(
                lambda: stores[0].storage_group.mutate(stores, callback),
                readback,
                background_tasks=self._background_tasks,
            )
            cancelled = cancelled or result.cancelled
            if result.state is DurableCommitState.COMMITTED:
                if result.value is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if cancelled:
                    raise asyncio.CancelledError
                return result.value
            if result.state is DurableCommitState.NOT_COMMITTED:
                if (
                    isinstance(result.error, AIError)
                    and result.error.code is ErrorCode.STORAGE_CONFLICT
                ):
                    await asyncio.sleep(0)
                    continue
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from result.error
            if isinstance(result.error, AIError) and result.error.code in {
                ErrorCode.TOOL_OPERATION_CONFLICT,
                ErrorCode.TOOL_RESULT_CONFLICT,
                ErrorCode.TOOL_EFFECT_UNKNOWN,
            }:
                raise result.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from result.error


__all__ = ["RuntimeStateCommands"]
