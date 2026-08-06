#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Budget reservation state values."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReservationState(StrEnum):
    """Budget reservation lifecycle."""

    RESERVED = "RESERVED"
    SETTLED = "SETTLED"
    UNKNOWN_RESERVED = "UNKNOWN_RESERVED"


class UsageBudget(BaseModel):
    """Pure aggregate budget admission value."""

    model_config = ConfigDict(frozen=True)

    budget_id: str
    max_cost_microusd: int = Field(ge=0)
    reserved_cost_microusd: int = Field(default=0, ge=0)

    def can_reserve(self, amount_microusd: int) -> bool:
        """Return whether an atomic reservation can fit the cap."""
        return amount_microusd >= 0 and self.reserved_cost_microusd + amount_microusd <= self.max_cost_microusd


class AgentRunBudgetPlan(BaseModel):
    """Conservative per-run reservation plan."""

    model_config = ConfigDict(frozen=True)

    reservation_id: str
    execution_id: str
    run_id: str
    model_route_id: str
    request_limit: int = Field(ge=1)
    provider_attempt_limit: int = Field(ge=1)
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    max_total_tokens: int = Field(ge=1)
    max_cost_microusd: int = Field(ge=0)
    price_table_version: str
    budget_id: str = "execution"
    max_reserved_cost_microusd: int = Field(default=0, ge=0)


class UsageReservation(BaseModel):
    """Mutable only through explicit state-transition methods."""

    model_config = ConfigDict(frozen=True)

    reservation_id: str
    state: ReservationState = ReservationState.RESERVED
    confirmed_cost_microusd: int = 0
    budget_id: str = "execution"
    reserved_cost_microusd: int = 0

    def settle(self, cost_microusd: int) -> "UsageReservation":
        """Settle a reservation with confirmed usage."""
        if cost_microusd < 0:
            raise ValueError("cost must be non-negative")
        return self.model_copy(
            update={
                "state": ReservationState.SETTLED,
                "confirmed_cost_microusd": cost_microusd,
            }
        )

    def mark_unknown(self) -> "UsageReservation":
        """Retain conservative funds after an unconfirmed attempt."""
        return self.model_copy(update={"state": ReservationState.UNKNOWN_RESERVED})
