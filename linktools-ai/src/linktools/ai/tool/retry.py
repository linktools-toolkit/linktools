#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RetryPolicy: decides whether a failed tool call should be retried. The
default policy retries ONLY clearly-transient errors and never retries a
mutating non-idempotent tool (a half-applied write retried blind is a hazard).
Backoff is applied between attempts."""

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..errors import (
    CapabilityResolutionError,
    IdempotencyConflictError,
    PipelineExecutionError,
    ToolApprovalRequiredError,
    ToolDeniedError,
    ToolPolicyResolutionError,
    ToolSchemaValidationError,
    TransientToolError,
)

if TYPE_CHECKING:
    from ..tool.models import ToolDescriptor
    from .policy import EffectiveToolPolicy

# Errors that can NEVER succeed on retry -- retrying them is wasted work and,
# for mutating tools, dangerous. Subclasses of these are covered too.
_PERMANENT: "tuple[type[BaseException], ...]" = (
    ToolDeniedError,
    ToolApprovalRequiredError,
    ToolSchemaValidationError,
    ToolPolicyResolutionError,
    PipelineExecutionError,
    CapabilityResolutionError,
    IdempotencyConflictError,
    ValueError,
    TypeError,
    PermissionError,
    KeyError,
    LookupError,
)


@runtime_checkable
class RetryPolicy(Protocol):
    def should_retry(
        self,
        *,
        error: BaseException,
        attempt: int,
        policy: "EffectiveToolPolicy",
        descriptor: "ToolDescriptor",
    ) -> bool: ...


def backoff_delay(attempt: int, base: float = 0.1, cap: float = 5.0) -> float:
    """Exponential backoff (base * 2**(attempt-1)) capped at ``cap`` seconds."""
    return min(cap, base * (2 ** max(0, attempt - 1)))


class DefaultRetryPolicy:
    """Retries only clearly-transient errors; never retries a mutating
    non-idempotent tool regardless of error type."""

    def should_retry(
        self,
        *,
        error: BaseException,
        attempt: int,
        policy: "EffectiveToolPolicy",
        descriptor: "ToolDescriptor",
    ) -> bool:
        # A mutating, non-idempotent tool may have partially applied its effect
        # -- never retry it blind, no matter the error.
        if descriptor.mutating and not policy.idempotent:
            return False
        if isinstance(error, _PERMANENT):
            return False
        if isinstance(error, TransientToolError):
            return True
        # A timeout MIGHT be transient, but only for read-only / idempotent
        # tools (covered above for mutating). asyncio.TimeoutError is retried.
        if isinstance(error, asyncio.TimeoutError):
            return True
        return False
