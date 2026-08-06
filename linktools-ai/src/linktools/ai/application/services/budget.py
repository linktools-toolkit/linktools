#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Budget reservation and reconciliation coordination."""

from ...domain.budget import AgentRunBudgetPlan, UsageReservation
from ...foundation.errors import ErrorCode, LinktoolsAIError


class BudgetService:
    def __init__(self, repository: object) -> None:
        self._repository = repository

    async def reserve(self, plan: AgentRunBudgetPlan) -> UsageReservation:
        return await self._repository.reserve(plan)

    async def reconcile(self, reservation: UsageReservation, cost_microusd: int) -> UsageReservation:
        settled = reservation.settle(cost_microusd)
        return await self._repository.reconcile(settled)

    async def mark_unknown(self, reservation_id: str) -> UsageReservation:
        return await self._repository.mark_unknown(reservation_id)


__all__ = ["BudgetService"]
