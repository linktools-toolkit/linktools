#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strongly-typed event payloads, per docs/linktools-ai.md section 23.2. Each
payload carries the minimum data meaningful for that event type -- the spec
mandates which payload TYPES must exist, not their exact fields."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RunStarted:
    run_id: str
    runnable_id: str


@dataclass(frozen=True, slots=True)
class RunCompleted:
    run_id: str
    result_summary: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunFailed:
    run_id: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class RunPaused:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class RunResumed:
    run_id: str


@dataclass(frozen=True, slots=True)
class RunCancelled:
    run_id: str
    reason: "str | None" = None


@dataclass(frozen=True, slots=True)
class ModelStarted:
    model_type: str


@dataclass(frozen=True, slots=True)
class ModelCompleted:
    model_type: str
    token_usage: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelFailed:
    model_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    tool_name: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    tool_name: str
    tool_call_id: str
    success: bool


@dataclass(frozen=True, slots=True)
class ToolFailed:
    tool_name: str
    tool_call_id: str
    error_message: str
