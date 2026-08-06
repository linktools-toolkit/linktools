#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Approval request and decision coordination."""

from ...domain.approval import PendingDeferredCall
from ...domain.execution import ApprovalDecisionRequest, ApprovalDecisionResult
from ...foundation.errors import ErrorCode, LinktoolsAIError


class ApprovalService:
    def __init__(self, repository: object, gateway: "object | None" = None) -> None:
        self._repository = repository
        self._gateway = gateway

    async def request(self, call: PendingDeferredCall) -> PendingDeferredCall:
        await self._repository.create_pending(call)
        return call

    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult:
        result = await self._repository.record_decision(request)
        if self._gateway is not None:
            await self._gateway.update(execution_id, "approval", request)
        return result


__all__ = ["ApprovalService"]
