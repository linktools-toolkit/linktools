#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Content-retention projection for native SDK model attempt diagnostics."""

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from linktools.core import environ
from pydantic_ai import messages as sdk_messages
from pydantic_ai.messages import ModelResponse

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelRequestAttempt

_MODEL_REQUEST_ATTEMPT = getattr(sdk_messages, "ModelRequestAttempt", None)


def project_model_attempts(
    attempts: Sequence[object], *, content_omission_reason: str | None = None,
) -> "list[ModelRequestAttempt]":
    """Copy only changed diagnostics while preserving SDK attempt usage and identity."""
    if content_omission_reason is not None and (
        not isinstance(content_omission_reason, str) or not content_omission_reason.strip()
    ):
        raise ValueError("content omission reason must be nonempty")
    projected = []
    for value in attempts:
        if _MODEL_REQUEST_ATTEMPT is None or not isinstance(value, _MODEL_REQUEST_ATTEMPT):
            raise TypeError("model request attempt is invalid")
        attempt = cast("ModelRequestAttempt", value)
        error = None if content_omission_reason is not None else (
            environ.logging.redact(attempt.error) if attempt.error is not None else None
        )
        projected.append(replace(attempt, error=error) if error != attempt.error else attempt)
    return projected


def project_model_response_attempts(
    response: ModelResponse, *, content_omission_reason: str | None = None,
) -> ModelResponse:
    """Apply request retention without mutating provider responses or usage."""
    attempts = getattr(response, "failed_attempts", None)
    projected = project_model_attempts(attempts or (), content_omission_reason=content_omission_reason)
    return replace(response, failed_attempts=projected) if attempts is not None and projected != attempts else response


__all__ = ["project_model_attempts", "project_model_response_attempts"]
