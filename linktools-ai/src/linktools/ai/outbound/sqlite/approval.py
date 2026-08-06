#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"SQLite approval persistence adapter boundary."


class ApprovalStore:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create_pending(self, call: object) -> object:
        return await self._operation.create_pending(call)

    async def record_decision(self, decision: object) -> object:
        return await self._operation.record_decision(decision)

    async def get_decision(self, decision_id: str) -> "object | None":
        return await self._operation.get_decision(decision_id)

    async def list_pending(self, execution_id: str) -> "tuple[object, ...]":
        return await self._operation.list_pending(execution_id)


__all__ = ["ApprovalStore"]
