#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private local service coordination for launch and stream leases."""

from collections.abc import AsyncIterator

from ..core import Principal
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
        agent_digest: str,
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
                agent_digest,
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
        if lease is None:
            async for event in self._event.stream(
                execution_id,
                principal=principal,
                after_sequence=after_sequence,
            ):
                yield event
            return
        try:
            async for event in self._event._stream_claimed(
                lease,
                principal=principal,
                after_sequence=after_sequence,
            ):
                yield event
        finally:
            if lease.state == "PREPARED":
                self._abort(lease)


    def _abort(self, lease: _PreparedStreamLease) -> None:
        self._leases.pop(lease.execution_id, None)
        self._event.live_broker.abort_local_producer(lease)


__all__ = ["_LocalRuntimeCoordinator"]
