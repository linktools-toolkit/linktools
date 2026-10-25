#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Request-local streaming attempt observations, independent of HTTP dispatches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage


@dataclass(frozen=True, slots=True)
class ModelStreamAttempt:
    """A transient observation; response content is not safe error metadata.

    The hosting Runtime owns retention and publication. HTTP positions are a
    half-open range in its request recorder, never an attempt identity.
    """

    stream_attempt_seq: int
    status: Literal["RUNNING", "SUCCEEDED", "FAILED", "CANCELLED"]
    started_at: datetime
    finished_at: datetime | None
    duration_ns: int | None
    client_request_start: int
    client_request_end: int
    response: ModelResponse | None
    usage: RequestUsage | None
    error: BaseException | None
    retry_scheduled: bool
    replaces_stream_attempt_seq: int | None = None


__all__ = ["ModelStreamAttempt"]
