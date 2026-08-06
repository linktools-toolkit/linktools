#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Temporal Activity boundary for budget reconciliation."""


class ReconcileAgentRunBudgetActivity:
    def __init__(self, service: object) -> None:
        self._service = service

    async def execute(self, request: object) -> object:
        return await self._service.reconcile(request)


__all__ = ["ReconcileAgentRunBudgetActivity"]
