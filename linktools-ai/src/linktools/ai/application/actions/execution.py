#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution actions over injected ports."""

from linktools.core import environ

from ...domain.execution import ExecutionHandle, ExecutionProfile, ExecutionRequest, ExecutionStatus, ExecutionView
from ...foundation.digest import hmac_digest, sha256_digest
from ...foundation.errors import ErrorCode, LinktoolsAIError
from ...foundation.ids import workflow_id
from ...foundation.json import canonical_json_bytes

logger = environ.get_logger("ai.application.actions.execution")


class StartExecution:
    """Create an idempotent execution projection and start its gateway."""

    def __init__(self, repository: object, gateway: object, identity_key: bytes, profile_policy: object, tenant_id: str = "authenticated") -> None:
        self._repository = repository
        self._gateway = gateway
        self._identity_key = identity_key
        self._profile_policy = profile_policy
        self._tenant_id = tenant_id

    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        input_bytes = canonical_json_bytes(request.input)
        if len(input_bytes) > 256 * 1024:
            raise LinktoolsAIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE, "input exceeds inline limit")
        if len(canonical_json_bytes(request.metadata)) > 16 * 1024:
            raise LinktoolsAIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE, "metadata exceeds inline limit")
        execution_id = hmac_digest(self._identity_key, request.idempotency_key.encode("utf-8"))[:32]
        workflow = workflow_id(self._tenant_id, execution_id)
        fingerprint = sha256_digest(canonical_json_bytes(request))
        existing = await self._repository.get(execution_id)
        if existing is not None:
            if existing.client_request_fingerprint not in (None, fingerprint):
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT, "request identity differs")
            if existing.status is ExecutionStatus.ACCEPTED:
                started = await self._gateway.start(request, execution_id, existing.workflow_id or workflow)
                if isinstance(started, ExecutionHandle):
                    return started
            return ExecutionHandle(execution_id=execution_id, workflow_id=existing.workflow_id, status=existing.status)
        release = await self._repository.resolve_release(request.agent_id, request.agent_revision)
        if release is None:
            raise LinktoolsAIError(ErrorCode.AGENT_NOT_FOUND, "agent release not found")
        profile = self._profile_policy.select(request, release)
        view = ExecutionView(execution_id=execution_id, tenant_id=self._tenant_id, agent_id=release.agent_id, agent_revision=release.revision, profile=profile, conversation_id=request.conversation_id, status=ExecutionStatus.ACCEPTED, workflow_id=workflow, client_request_fingerprint=fingerprint)
        await self._repository.upsert_projection(view)
        logger.info("execution start id=%s profile=%s", execution_id, profile.value)
        started = await self._gateway.start(request, execution_id, workflow)
        if isinstance(started, ExecutionHandle):
            return started
        return ExecutionHandle(execution_id=execution_id, workflow_id=workflow, status=ExecutionStatus.ACCEPTED)


class InspectExecution:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, execution_id: str) -> object: return await self._repository.get(execution_id)


class GetExecutionResult:
    def __init__(self, repository: object) -> None: self._repository = repository
    async def execute(self, execution_id: str) -> object: return await self._repository.get(execution_id)


class CancelExecution:
    def __init__(self, gateway: object) -> None: self._gateway = gateway
    async def execute(self, execution_id: str, request: object) -> object: return await self._gateway.cancel(execution_id, request)


class RetryExecution:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, execution_id: str, request: object) -> object: return await self._service.retry(execution_id, request)


class ForkExecution:
    def __init__(self, service: object) -> None: self._service = service
    async def execute(self, execution_id: str, request: object) -> object: return await self._service.fork(execution_id, request)


__all__ = ["CancelExecution", "ForkExecution", "GetExecutionResult", "InspectExecution", "RetryExecution", "StartExecution"]
