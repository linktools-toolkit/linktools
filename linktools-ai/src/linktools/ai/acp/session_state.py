#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Session operation ownership and invariant checks."""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import uuid4

from .errors import internal_error, request_error

if TYPE_CHECKING:
    from .session_models import ActiveAcpSession


logger = logging.getLogger("linktools.ai.acp.session_state")


class SessionOperationKind(str, Enum):
    PROMPT = "prompt"
    LOAD = "load"
    RESUME = "resume"
    FORK = "fork"
    SET_MODE = "set_mode"
    SET_CONFIG = "set_config"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class SessionOperationToken:
    operation_id: str
    kind: SessionOperationKind
    epoch: int
    owner_task: "asyncio.Task[Any]"
    done: asyncio.Event


class SessionOperationCoordinator:
    """Own the single primary operation allowed to mutate one session."""

    async def reserve(
        self,
        active: "ActiveAcpSession",
        kind: SessionOperationKind,
        *,
        execution_id: "str | None" = None,
    ) -> SessionOperationToken:
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise internal_error("session_operation_owner_missing")
        if kind is SessionOperationKind.PROMPT and execution_id is None:
            raise internal_error("prompt_execution_id_missing")
        async with active.lock:
            if active.record.closed:
                raise request_error("session_closed", session_id=active.record.session_id)
            if active.cleanup_required:
                raise request_error(
                    "session_cleanup_required",
                    session_id=active.record.session_id,
                    details={"operation": "cleanup"},
                )
            if active.closing_requested:
                raise request_error("session_closing", session_id=active.record.session_id)
            if active.operation is not None:
                raise request_error(
                    "session_busy",
                    session_id=active.record.session_id,
                    details={"operation": active.operation.kind.value},
                )
            active.operation_epoch += 1
            token = SessionOperationToken(
                operation_id=uuid4().hex,
                kind=kind,
                epoch=active.operation_epoch,
                owner_task=owner_task,
                done=asyncio.Event(),
            )
            active.operation = token
            if kind is SessionOperationKind.PROMPT:
                active.active_execution_id = execution_id
            session_id = active.record.session_id
        logger.info(
            "event=acp.session.operation_reserved session_id=%s operation_id=%s kind=%s epoch=%s execution_id=%s",
            session_id,
            token.operation_id,
            token.kind.value,
            token.epoch,
            execution_id,
        )
        return token

    async def release(
        self,
        active: "ActiveAcpSession",
        token: SessionOperationToken,
    ) -> None:
        async with active.lock:
            if active.operation == token:
                active.operation = None
                if token.kind is SessionOperationKind.PROMPT:
                    active.active_execution_id = None
            token.done.set()
            session_id = active.record.session_id
        logger.info(
            "event=acp.session.operation_released session_id=%s operation_id=%s kind=%s epoch=%s",
            session_id,
            token.operation_id,
            token.kind.value,
            token.epoch,
        )

    async def validate(
        self,
        active: "ActiveAcpSession",
        token: SessionOperationToken,
    ) -> bool:
        async with active.lock:
            return (
                active.operation == token
                and (token.kind is SessionOperationKind.CLOSE or not active.cleanup_required)
                and not active.record.closed
                and active.operation_epoch == token.epoch
                and not token.owner_task.done()
            )

    async def request_close(
        self,
        active: "ActiveAcpSession",
        close_factory: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task[Any]:
        async with active.lock:
            if active.close_task is not None:
                task = active.close_task
                joined = True
            else:
                active.closing_requested = True
                task = asyncio.create_task(close_factory())
                active.close_task = task
                joined = False
            session_id = active.record.session_id
        if joined:
            logger.info(
                "event=acp.session.close_joined_existing_task session_id=%s",
                session_id,
            )
        return task

    async def reserve_close(
        self,
        active: "ActiveAcpSession",
    ) -> SessionOperationToken:
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise internal_error("session_operation_owner_missing")
        async with active.lock:
            if active.operation is not None:
                raise request_error(
                    "session_busy",
                    session_id=active.record.session_id,
                    details={"operation": active.operation.kind.value},
                )
            active.operation_epoch += 1
            token = SessionOperationToken(
                operation_id=uuid4().hex,
                kind=SessionOperationKind.CLOSE,
                epoch=active.operation_epoch,
                owner_task=owner_task,
                done=asyncio.Event(),
            )
            active.operation = token
            return token

    async def clear_close_task(
        self,
        active: "ActiveAcpSession",
        task: asyncio.Task[Any],
    ) -> None:
        async with active.lock:
            if active.close_task is task:
                active.close_task = None


def assert_session_invariants(active: "ActiveAcpSession") -> None:
    if (active.pending_permission is None) != (active.pending_permission_task is None):
        raise AssertionError("permission token/task ownership is split")
    if active.record.closed and active.active_execution_id is not None:
        raise AssertionError("closed session owns an execution")
    if active.record.closed and active.operation is not None:
        raise AssertionError("closed session owns an operation")
    if active.cleanup_required and active.record.closed:
        raise AssertionError("cleanup_required session is closed")
    if active.close_task is not None and not active.closing_requested:
        raise AssertionError("close task exists without close request")
    if active.operation is not None:
        if active.operation.kind is SessionOperationKind.PROMPT:
            if active.active_execution_id is None:
                raise AssertionError("prompt operation has no execution")
        elif active.active_execution_id is not None:
            raise AssertionError("non-prompt operation owns an execution")
    if not set(active.terminal_release_tasks).issubset(active.terminal_handles):
        raise AssertionError("terminal release task has no terminal owner")


__all__ = [
    "SessionOperationCoordinator",
    "SessionOperationKind",
    "SessionOperationToken",
    "assert_session_invariants",
]
