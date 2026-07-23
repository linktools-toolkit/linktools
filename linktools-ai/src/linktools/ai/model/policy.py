#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModelPolicy: primary model + fallback chain + limits.

The registry parser validates these fields, but a custom provider can construct
a ModelPolicy directly -- so the model itself enforces the same contract
(non-empty primary, non-negative int request_retries, positive-finite timeout,
positive int max_tokens, finite non-negative Decimal budget)."""

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    primary: str
    fallbacks: "tuple[str, ...]" = ()
    request_retries: int = 0
    timeout_seconds: "float | None" = None
    max_tokens: "int | None" = None
    budget: "Decimal | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.primary, str) or not self.primary.strip():
            raise ValueError("ModelPolicy.primary must be a non-empty string")
        if not isinstance(self.fallbacks, tuple):
            raise TypeError("ModelPolicy.fallbacks must be a tuple")
        for item in self.fallbacks:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("ModelPolicy.fallbacks must be non-empty strings")
        if (
            isinstance(self.request_retries, bool)
            or not isinstance(self.request_retries, int)
            or self.request_retries < 0
        ):
            raise ValueError("ModelPolicy.request_retries must be a non-negative integer")
        if self.timeout_seconds is not None:
            if (
                isinstance(self.timeout_seconds, bool)
                or not isinstance(self.timeout_seconds, (int, float))
                or not math.isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0
            ):
                raise ValueError(
                    "ModelPolicy.timeout_seconds must be a positive finite number"
                )
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if self.max_tokens is not None:
            if (
                isinstance(self.max_tokens, bool)
                or not isinstance(self.max_tokens, int)
                or self.max_tokens <= 0
            ):
                raise ValueError("ModelPolicy.max_tokens must be a positive integer")
        if self.budget is not None:
            if not isinstance(self.budget, Decimal):
                raise TypeError("ModelPolicy.budget must be a Decimal or None")
            if not self.budget.is_finite() or self.budget < 0:
                raise ValueError("ModelPolicy.budget must be finite and non-negative")
