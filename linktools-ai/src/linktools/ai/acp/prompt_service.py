#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP prompt execution and permission lifecycle."""

import asyncio
import logging
from typing import Any

from ..execution.domain import ApprovalDecision, RunStatus
from ..execution.live_events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEventHub,
    ExecutionFailed,
    ExecutionPaused,
)
from ..governance.identity import PrincipalContext
from ..runtime.facade import Runtime
from .errors import internal_error
from .event_mapper import AcpEventMapper
from .execution import AcpExecutionAdapter
from .session_models import ActiveAcpSession, PendingPermissionToken
from .session_state import SessionOperationKind
from .task_utils import cancel_and_wait


logger = logging.getLogger("linktools.ai.acp.prompt_service")


class AcpPromptService:
    """Own Runtime execution, event forwarding, and permission decisions."""

    def __init__(
        self,
        *,
        runtime: Runtime,
        event_hub: ExecutionEventHub,
        principal: PrincipalContext,
    ) -> None:
        self.runtime = runtime
        self.event_hub = event_hub
        self.principal = principal
        self.event_mapper = AcpEventMapper()
        self._connection: Any = None

    def set_connection(self, connection: Any) -> None:
        self._connection = connection

    async def clear_permission(self, active: ActiveAcpSession, execution_id: str) -> None:
        async with active.lock:
            token = active.pending_permission
            if token is None or token.execution_id != execution_id:
                return
            task = active.pending_permission_task
            active.pending_permission = None
            active.pending_permission_task = None
        if task is not None:
            await cancel_and_wait(task, timeout=1)

    async def execute(
        self,
        active: ActiveAcpSession,
        execution_id: str,
        spec: Any,
        prompt: Any,
    ) -> Any:
        import acp.schema as schema

        operation = "run"
        while True:
            subscription = await self.event_hub.subscribe(execution_id)
            task: "asyncio.Task[Any] | None" = None
            try:
                toolsets = await active.mcp_resources.toolsets()
                task = asyncio.create_task(
                    self.runtime.run(
                        spec,
                        prompt,
                        principal=self.principal,
                        session_id=active.record.session_id,
                        execution_id=execution_id,
                        extra_toolsets=toolsets,
                    )
                    if operation == "run"
                    else self.runtime.resume(
                        execution_id,
                        principal=self.principal,
                        extra_toolsets=toolsets,
                    )
                )
                await self._drain_operation(active, execution_id, subscription, task)
                await task
            finally:
                if task is not None and not task.done():
                    task.cancel()
                    await cancel_and_wait(task, timeout=5)
                await subscription.release()
            record = await self.runtime.get_execution_record(
                execution_id,
                principal=self.principal,
            )
            if record is None:
                raise internal_error(
                    "execution_record_missing",
                    session_id=active.record.session_id,
                    execution_id=execution_id,
                )
            if record.status is RunStatus.PAUSED:
                decision = await self.request_permission(active, record)
                if decision is ApprovalDecision.ALLOW:
                    operation = "resume"
                    continue
                return schema.PromptResponse(stopReason="cancelled")
            if record.status is RunStatus.CANCELLED:
                return schema.PromptResponse(stopReason="cancelled")
            if record.status is RunStatus.FAILED:
                raise internal_error(
                    "execution_failed",
                    session_id=active.record.session_id,
                    execution_id=execution_id,
                )
            return schema.PromptResponse(
                stopReason=AcpExecutionAdapter.stop_reason(record.status)
            )

    async def _drain_operation(
        self,
        active: ActiveAcpSession,
        execution_id: str,
        subscription: Any,
        task: "asyncio.Task[Any]",
    ) -> None:
        event_task = asyncio.create_task(subscription.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    event = event_task.result()
                    await self._publish_update(active, event)
                    if isinstance(
                        event,
                        (ExecutionPaused, ExecutionCompleted, ExecutionFailed, ExecutionCancelled),
                    ):
                        return
                    event_task = asyncio.create_task(subscription.__anext__())
                    continue
                task_error = task.exception()
                if task_error is not None:
                    raise task_error
                try:
                    event = await asyncio.wait_for(event_task, timeout=5)
                except asyncio.TimeoutError as exc:
                    raise internal_error(
                        "execution_event_missing",
                        session_id=active.record.session_id,
                        execution_id=execution_id,
                    ) from exc
                await self._publish_update(active, event)
                if isinstance(
                    event,
                    (ExecutionPaused, ExecutionCompleted, ExecutionFailed, ExecutionCancelled),
                ):
                    return
                event_task = asyncio.create_task(subscription.__anext__())
        finally:
            if not event_task.done():
                await cancel_and_wait(event_task, timeout=5)

    async def _publish_update(self, active: ActiveAcpSession, event: Any) -> None:
        update = self.event_mapper.map(event)
        if update is not None and self._connection is not None:
            await self._connection.session_update(active.record.session_id, update)

    async def request_permission(
        self,
        active: ActiveAcpSession,
        record: Any,
    ) -> "ApprovalDecision | None":
        if self._connection is None:
            raise internal_error(
                "client_connection_missing",
                session_id=active.record.session_id,
                execution_id=record.id,
            )
        import acp.schema as schema

        detail = await self.runtime.inspect(run_id=record.id, principal=self.principal)
        tool = detail.tool_calls[-1] if detail is not None and detail.tool_calls else None
        tool_call = schema.ToolCallUpdate(
            toolCallId=record.approval.tool_call_id if record.approval else "",
            title=record.approval.tool_name if record.approval else "approval",
            status="pending",
            rawInput=tool.arguments if tool is not None else None,
        )
        options = [
            schema.PermissionOption(optionId="allow_once", name="Allow once", kind="allow_once"),
            schema.PermissionOption(optionId="reject_once", name="Reject once", kind="reject_once"),
        ]
        request = self._connection.request_permission(
            active.record.session_id,
            tool_call,
            options,
        )
        async with active.lock:
            operation = active.operation
            if (
                active.active_execution_id != record.id
                or active.closing_requested
                or record.approval is None
                or (
                    operation is not None
                    and operation.kind is not SessionOperationKind.PROMPT
                )
            ):
                request.close()
                return None
            token = PendingPermissionToken(
                session_id=active.record.session_id,
                execution_id=record.id,
                approval_id=record.approval.approval_id,
                tool_call_id=record.approval.tool_call_id,
                epoch=active.operation_epoch,
            )
            task = asyncio.create_task(request)
            active.pending_permission = token
            active.pending_permission_task = task
        try:
            response = await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.info(
                "event=acp.permission.task_cancelled session_id=%s execution_id=%s operation_id=%s epoch=%s",
                token.session_id,
                token.execution_id,
                operation.operation_id if operation is not None else "legacy-recovery",
                token.epoch,
            )
            await cancel_and_wait(task, timeout=1)
            await self._cancel_permission_token(active, token)
            return None
        except Exception:
            await self._cancel_permission_token(active, token)
            return None
        outcome = response.outcome
        option_id = getattr(outcome, "option_id", None)
        decision = (
            ApprovalDecision.ALLOW
            if option_id == "allow_once"
            else ApprovalDecision.DENY
            if option_id == "reject_once"
            else None
        )
        current = await self.runtime.get_execution_record(
            token.execution_id,
            principal=self.principal,
        )
        async with active.lock:
            valid = (
                not active.closing_requested
                and active.active_execution_id == token.execution_id
                and active.pending_permission == token
                and active.operation_epoch == token.epoch
                and active.operation == operation
                and current is not None
                and current.status is RunStatus.PAUSED
                and current.approval is not None
                and current.approval.approval_id == token.approval_id
            )
            if active.pending_permission == token:
                active.pending_permission = None
                if active.pending_permission_task is task:
                    active.pending_permission_task = None
        if not valid:
            logger.info(
                "event=acp.permission.stale_result session_id=%s execution_id=%s approval_id=%s operation_epoch=%s",
                token.session_id,
                token.execution_id,
                token.approval_id,
                token.epoch,
            )
            return None
        if decision is None:
            await self.runtime.cancel(record.id, principal=self.principal)
            return None
        await self.runtime.decide_approval(
            record.id,
            approval_id=token.approval_id,
            decision=decision,
            principal=self.principal,
        )
        return decision

    async def _cancel_permission_token(
        self,
        active: ActiveAcpSession,
        token: PendingPermissionToken,
    ) -> None:
        async with active.lock:
            if active.pending_permission != token or active.active_execution_id != token.execution_id:
                return
            active.pending_permission = None
            task = active.pending_permission_task
            active.pending_permission_task = None
            execution_id = active.active_execution_id
        if task is not None and not task.done():
            await cancel_and_wait(task, timeout=1)
        try:
            await asyncio.wait_for(
                self.runtime.cancel(execution_id, principal=self.principal),
                timeout=5,
            )
        except Exception:
            logger.debug("ACP permission cancellation failed execution=%s", execution_id)


__all__ = ["AcpPromptService"]
