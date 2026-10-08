#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable dispatch identities underlying the budget usage projection."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class BudgetModelReservation:
    scope_id: str
    request_id: str
    owner_id: str
    admission_id: str
    status: Literal["in_flight", "settled", "unknown"] = "in_flight"
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.scope_id, self.request_id, self.owner_id, self.admission_id):
            if not isinstance(value, str) or not value:
                raise ValueError("model budget reservation identities are required")
        if self.status not in {"in_flight", "settled", "unknown"}:
            raise ValueError("model budget reservation status is invalid")
        if self.status == "settled":
            if isinstance(self.total_tokens, bool) or not isinstance(self.total_tokens, int) or self.total_tokens < 0:
                raise ValueError("settled model tokens must be a non-negative integer")
        elif self.total_tokens is not None:
            raise ValueError("unresolved model usage cannot carry a known token count")


@dataclass(frozen=True, slots=True)
class BudgetToolReservation:
    scope_id: str
    call_id: str
    admission_id: str

    def __post_init__(self) -> None:
        for value in (self.scope_id, self.call_id, self.admission_id):
            if not isinstance(value, str) or not value:
                raise ValueError("tool budget reservation identities are required")


__all__ = ["BudgetModelReservation", "BudgetToolReservation"]
