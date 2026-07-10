#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelPolicy: primary model + fallback chain + limits."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    primary: str
    fallbacks: "tuple[str, ...]" = ()
    max_retries: int = 0
    timeout_seconds: "float | None" = None
    max_tokens: "int | None" = None
    budget: "Decimal | None" = None
