#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Public runtime facade over execution orchestration and authorized queries."""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from linktools.core import environ

from ..agent.mcp.client import MCPConnectionPool
from ..agent.sandbox.protocols import Sandbox
from ..errors import PrincipalAccessDeniedError
from ..execution.live_events import ExecutionEventHub
from ..governance.identity import PrincipalContext
from .interaction import InteractiveRunService
from .session import ResourceFailure, RuntimeSessionService, SessionCloseResult

logger = environ.get_logger("ai.runtime.facade")

if TYPE_CHECKING:
    from typing import Any, Mapping

    from ..agent.spec import AgentSpec
    from ..agent.assembly.assembler import AgentAssembler
    from ..execution.domain import ApprovalDecision, RunRecord
    from ..execution.commands import ParentLeaseGuard
    from ..execution.service import ChildRunResult
    from ..execution.session import SessionContextSeed, SessionRecord
    from ..execution.query import (
        ExecutionQueryService,
        ExecutionDetailView,
        RunMessagesView,
    )
    from ..execution.service import ExecutionService
    from ..execution.swarm_service import SwarmExecutionService
    from ..json import JsonValue
    from .prompt import UserPrompt
    from ..tasks.models import TaskPlan
    from ..tasks.swarm.models import SwarmExecutionOutcome, SwarmRunView
    from ..tasks.swarm.spec import SwarmSpec


@dataclass(frozen=True, slots=True)
class RuntimeCloseResult:
    closed: bool
    session_results: "tuple[SessionCloseResult, ...]" = ()


@dataclass(frozen=True, slots=True)
class Runtime:
    execution: "ExecutionService"
    query: "ExecutionQueryService"
    assembler: "AgentAssembler"
    tool_execution_ready: bool
    sandbox: "Sandbox | None" = None
    mcp_connections: "MCPConnectionPool | None" = None
    execution_event_hub: "ExecutionEventHub | None" = None
    swarm: "SwarmExecutionService | None" = None
    sessions: "RuntimeSessionService | None" = None
    interactions: "InteractiveRunService | None" = None
    _close_task: "asyncio.Task[RuntimeCloseResult] | None" = field(
        default=None, init=False, repr=False, compare=False
    )
    _session_shutdown_task: "asyncio.Task[tuple[SessionCloseResult, ...]] | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    async def run(
        self,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        *,
        principal: PrincipalContext,
        session_id: "str | None" = None,
        execution_id: "str | None" = None,
        extra_toolsets: "tuple[object, ...]" = (),
    ) -> object:
        # Omitting a session ID creates an isolated single-turn session.
        if not isinstance(principal, PrincipalContext):
            raise PrincipalAccessDeniedError("a valid PrincipalContext is required")
        resolved_session_id = session_id or uuid4().hex
        resolved_execution_id = execution_id or uuid4().hex
        return await self.execution.run(
            spec,
            prompt,
            principal=principal,
            session_id=resolved_session_id,
            execution_id=resolved_execution_id,
            extra_toolsets=extra_toolsets,
        )

    async def resume(
        self,
        execution_id: str,
        *,
        principal: PrincipalContext,
        extra_toolsets: "tuple[object, ...]" = (),
    ) -> object:
        return await self.execution.resume(
            execution_id,
            principal=principal,
            extra_toolsets=extra_toolsets,
        )

    async def run_child(
        self,
        spec: "AgentSpec",
        prompt: str,
        *,
        principal: PrincipalContext,
        session_id: str,
        execution_id: str,
        root_execution_id: str,
        parent_execution_id: str,
        parent_guard: "ParentLeaseGuard",
        metadata: "Mapping[str, Any] | None" = None,
    ) -> "ChildRunResult":
        return await self.execution.run_child(
            spec,
            prompt,
            principal=principal,
            session_id=session_id,
            execution_id=execution_id,
            root_execution_id=root_execution_id,
            parent_execution_id=parent_execution_id,
            parent_guard=parent_guard,
            metadata=metadata,
        )

    async def fork_session(
        self,
        source_session_id: str,
        target_session_id: str,
        principal: PrincipalContext,
    ) -> "SessionRecord":
        return await self.execution.fork_session(
            source_session_id, target_session_id, principal
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: PrincipalContext,
        context_seed: "SessionContextSeed | None" = None,
    ) -> "SessionRecord":
        return await self.execution.create_session(
            session_id, principal=principal, context_seed=context_seed
        )

    async def get_session(
        self, session_id: str, *, principal: PrincipalContext
    ) -> "SessionRecord | None":
        return await self.execution.get_session(session_id, principal=principal)

    async def get_execution_record(
        self, execution_id: str, *, principal: PrincipalContext
    ) -> "RunRecord | None":
        return await self.execution.get_execution_record(execution_id, principal=principal)

    async def list_sessions(
        self, *, principal: PrincipalContext
    ) -> "tuple[SessionRecord, ...]":
        return await self.execution.list_sessions(principal=principal)

    async def cancel(self, execution_id: str, *, principal: PrincipalContext) -> None:
        await self.execution.cancel(execution_id, principal=principal)

    async def decide_approval(
        self,
        execution_id: str,
        *,
        approval_id: str,
        decision: "ApprovalDecision",
        principal: PrincipalContext,
    ) -> "RunRecord":
        return await self.execution.decide_approval(
            execution_id,
            approval_id=approval_id,
            decision=decision,
            principal=principal,
        )

    async def inspect(
        self, *, run_id: str, principal: PrincipalContext
    ) -> "ExecutionDetailView":
        return await self.query.get_run_detail(run_id=run_id, principal=principal)

    async def get_messages(
        self, *, run_id: str, principal: PrincipalContext
    ) -> "tuple[JsonValue, ...]":
        return await self.query.get_run_messages(run_id=run_id, principal=principal)

    async def get_session_messages(
        self, *, session_id: str, principal: PrincipalContext
    ) -> "tuple[RunMessagesView, ...]":
        return await self.query.get_session_messages(
            session_id=session_id, principal=principal
        )

    async def run_swarm(
        self,
        spec: "SwarmSpec",
        task_plan: "TaskPlan",
        *,
        principal: PrincipalContext,
        session_id: "str | None" = None,
        execution_id: "str | None" = None,
    ) -> "SwarmExecutionOutcome":
        if self.swarm is None:
            raise RuntimeError("swarm execution is not available on this runtime")
        return await self.swarm.run_swarm(
            spec,
            task_plan,
            principal=principal,
            session_id=session_id,
            execution_id=execution_id,
        )

    async def recover_swarm(
        self,
        execution_id: str,
        *,
        principal: PrincipalContext,
    ) -> "SwarmExecutionOutcome":
        if self.swarm is None:
            raise RuntimeError("swarm execution is not available on this runtime")
        return await self.swarm.recover_swarm(execution_id, principal=principal)

    async def inspect_swarm(
        self,
        execution_id: str,
        *,
        principal: PrincipalContext,
    ) -> "SwarmRunView":
        if self.swarm is None:
            raise RuntimeError("swarm execution is not available on this runtime")
        return await self.swarm.inspect_swarm(execution_id, principal=principal)

    async def aclose(self) -> RuntimeCloseResult:
        task = self._close_task
        joined = task is not None
        if task is None:
            # task-owner: runtime.close
            task = asyncio.create_task(self._close_once())
            object.__setattr__(self, "_close_task", task)
        if joined:
            logger.info("event=runtime.close_joined")
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except BaseException:
                pass
            raise
        finally:
            if task.done() and self._close_task is task:
                try:
                    retry = not task.result().closed
                except BaseException:
                    retry = True
                if retry:
                    object.__setattr__(self, "_close_task", None)
        return result

    async def shutdown(self) -> "tuple[SessionCloseResult, ...]":
        return await self.shutdown_sessions()

    async def shutdown_sessions(self) -> "tuple[SessionCloseResult, ...]":
        if self.sessions is None:
            return ()
        task = self._session_shutdown_task
        if task is None:
            task = asyncio.create_task(self.sessions.shutdown())
            object.__setattr__(self, "_session_shutdown_task", task)
        else:
            logger.info("event=runtime.sessions_shutdown_joined")
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        finally:
            if task.done() and self._session_shutdown_task is task:
                object.__setattr__(self, "_session_shutdown_task", None)
        logger.info(
            "event=runtime.sessions_shutdown_completed failure_count=%s",
            sum(len(item.failures) for item in result),
        )
        return result

    async def _close_once(self) -> RuntimeCloseResult:
        results: "tuple[SessionCloseResult, ...]" = ()
        if self.sessions is not None:
            try:
                results = await self.sessions.shutdown()
            except BaseException as exc:
                results = (
                    SessionCloseResult(
                        False, (ResourceFailure("session", None, type(exc).__name__),)
                    ),
                )
        if self.mcp_connections is not None:
            try:
                mcp_result = await self.mcp_connections.close()
            except BaseException as exc:
                results += (
                    SessionCloseResult(
                        False, (ResourceFailure("mcp", None, type(exc).__name__),)
                    ),
                )
            else:
                if mcp_result is not None and not mcp_result.closed:
                    results += (
                        SessionCloseResult(
                            False,
                            tuple(
                                ResourceFailure("mcp", item.server_id, item.error_id)
                                for item in mcp_result.failures
                            ),
                        ),
                    )
        if self.sandbox is not None:
            try:
                await self.sandbox.terminate()
            except BaseException as exc:
                results += (
                    SessionCloseResult(
                        False, (ResourceFailure("sandbox", None, type(exc).__name__),)
                    ),
                )
        result = RuntimeCloseResult(all(item.closed for item in results), results)
        logger.info("event=runtime.closed runtime_closed=%s", result.closed)
        return result

    async def __aenter__(self) -> "Runtime":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


__all__ = ["Runtime", "RuntimeCloseResult"]
