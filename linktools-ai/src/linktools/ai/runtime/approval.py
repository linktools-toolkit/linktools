#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval decision API."""

from typing import Protocol

from ..core import Principal
from .services import ApprovalDecisionRequest, ApprovalDecisionResult, ApprovalView


class ApprovalQueryApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]': ...


class ApprovalApi(ApprovalQueryApi, Protocol):
    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult: ...


__all__ = ["ApprovalApi", "ApprovalQueryApi"]
