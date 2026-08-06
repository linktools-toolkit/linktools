#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Atomic budget protocol."""

from typing import Protocol


class BudgetRepository(Protocol):
    async def create(self, plan: object) -> object: ...
    async def reserve(self, plan: object) -> object: ...
    async def reconcile(self, reservation: object) -> object: ...
    async def mark_unknown(self, reservation_id: str) -> object: ...
    async def get(self, reservation_id: str) -> "object | None": ...


__all__ = ["BudgetRepository"]
