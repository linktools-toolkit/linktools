#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private local service coordination for launch and stream leases."""

from collections.abc import AsyncIterator

from linktools.core import environ

from ..core import ExecutionEventType, Principal
from ._event import DefaultEventService, _PreparedStreamLease
from ._execution import DefaultExecutionService
from ._local import LocalExecutionBackend
from ._session import DefaultSessionService
from .service_api import (
    ExecutionHandle,
    ExecutionRequest,
    ExecutionStreamEvent,
    ResumeSessionRequest,
)

_logger = environ.get_logger("ai.runtime.coordinator")


class _LocalRuntimeCoordinator:
    """Own the concrete-only launch gate used by the local stream path."""

    def __init__(
        self,
        execution: DefaultExecutionService,
        session: DefaultSessionService,
        event: DefaultEventService,
        backend: LocalExecutionBackend,
    ) -> None:
        self._execution = execution
        self._session = session
        self._event = event
        self._backend = backend
        self._leases: dict[str, _PreparedStreamLease] = {}

    async def run(
        self,
        binding_digest: str,
        request: ExecutionRequest,
    ) -> ExecutionHandle:
        lease: _PreparedStreamLease | None = None

        async def gate(execution_id: str) -> None:
            nonlocal lease
            if self._backend.worker_installed(execution_id):
                return
            lease = await self._event._prepare_local_stream(execution_id)
            self._leases[execution_id] = lease

        try:
            handle = await self._execution._run_with_launch_gate(binding_digest, request, gate)
            if lease is not None:
                await self._event._authorize_stream(handle.execution_id, request.principal)
                if lease.base_sequence is not None:
                    await self._event._claim_local_stream(lease)
                else:
                    self._abort(lease)
                    lease = None
        except BaseException:
            if lease is not None:
                self._abort(lease)
            raise
        return handle

    async def resume(
        self,
        agent_id: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
    ) -> ExecutionHandle:
        lease: _PreparedStreamLease | None = None

        async def gate(execution_id: str) -> None:
            nonlocal lease
            if self._backend.worker_installed(execution_id):
                return
            lease = await self._event._prepare_local_stream(execution_id)
            self._leases[execution_id] = lease

        try:
            handle = await self._session._resume_with_launch_gate(
                agent_id,
                binding_digest,
                session_id,
                request,
                gate,
            )
            if lease is not None:
                await self._event._authorize_stream(handle.execution_id, request.principal)
                if lease.base_sequence is not None:
                    await self._event._claim_local_stream(lease)
                else:
                    self._abort(lease)
                    lease = None
        except BaseException:
            if lease is not None:
                self._abort(lease)
            raise
        return handle

    async def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        lease = self._leases.pop(execution_id, None)
        terminal_yielded = False
        try:
            events = (
                self._event.stream(
                    execution_id,
                    principal=principal,
                    after_sequence=after_sequence,
                )
                if lease is None
                else self._event._stream_claimed(
                    lease,
                    principal=principal,
                    after_sequence=after_sequence,
                )
            )
            async for event in events:
                if event.event_type in {
                    ExecutionEventType.EXECUTION_SUCCEEDED,
                    ExecutionEventType.EXECUTION_FAILED,
                    ExecutionEventType.EXECUTION_CANCELLED,
                }:
                    terminal_yielded = True
                yield event
        finally:
            if lease is not None and lease.state == "PREPARED":
                self._abort(lease)
            if terminal_yielded:
                _logger.debug(
                    "local stream yielded terminal event; requesting execution handoff: execution=%s",
                    execution_id,
                )
                await self._execution.request_terminal_handoff(
                    execution_id,
                    tenant_id=principal.tenant_id,
                )

    def abandon_stream(self, execution_id: str) -> None:
        """Release a prepared local stream when the caller never consumes it."""
        lease = self._leases.pop(execution_id, None)
        if lease is not None and lease.state == "PREPARED":
            self._event.live_broker.abort_local_producer(lease)

    def _abort(self, lease: _PreparedStreamLease) -> None:
        self._leases.pop(lease.execution_id, None)
        self._event.live_broker.abort_local_producer(lease)


__all__ = ["_LocalRuntimeCoordinator"]
