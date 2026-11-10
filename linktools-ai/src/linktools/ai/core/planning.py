#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral plan owner and item values."""

from dataclasses import dataclass
from typing import Literal

PlanOwnerKind = Literal["session", "execution"]
PlanStatus = Literal["pending", "in_progress", "completed", "cancelled"]


@dataclass(frozen=True, slots=True)
class PlanItem:
    content: str
    status: PlanStatus = "pending"



__all__ = ["PlanItem", "PlanOwnerKind", "PlanStatus"]
