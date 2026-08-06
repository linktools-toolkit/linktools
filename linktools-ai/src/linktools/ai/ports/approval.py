#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Approval persistence protocol."""

from typing import Protocol


class ApprovalRepository(Protocol):
    async def create_pending(self, call: object) -> object: ...
    async def record_decision(self, decision: object) -> object: ...
    async def get_decision(self, decision_id: str) -> "object | None": ...
    async def list_pending(self, execution_id: str) -> "tuple[object, ...]": ...


__all__ = ["ApprovalRepository"]
