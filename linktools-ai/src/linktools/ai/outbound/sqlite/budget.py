#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"SQLite budget persistence adapter boundary."


class BudgetStore:
    def __init__(self, operation: object) -> None:
        self._operation = operation

    async def create(self, plan: object) -> object:
        return await self._operation.create(plan)

    async def reserve(self, plan: object) -> object:
        return await self._operation.reserve(plan)

    async def reconcile(self, reservation: object) -> object:
        return await self._operation.reconcile(reservation)

    async def mark_unknown(self, reservation_id: str) -> object:
        return await self._operation.mark_unknown(reservation_id)

    async def get(self, reservation_id: str) -> "object | None":
        return await self._operation.get(reservation_id)


__all__ = ["BudgetStore"]
