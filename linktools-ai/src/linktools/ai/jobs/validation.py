#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Payload and input validation.

Enforces size/depth/count limits before a task or signal enters the store, so
malicious or runaway handlers cannot exhaust resources. Returns normally if
valid; raises ValueError with a clear message on violation.
"""

import json

MAX_HANDLER_NAME = 255
MAX_TASK_KEY = 255
MAX_METADATA_BYTES = 256 * 1024
MAX_METADATA_DEPTH = 10
MAX_COMMANDS = 100
MAX_CHILD_TASKS = 100
MAX_OUTPUT_PAYLOAD_BYTES = 1024 * 1024


def validate_handler_name(name: str) -> None:
    if not name or len(name) > MAX_HANDLER_NAME:
        raise ValueError(
            f"handler name must be 1..{MAX_HANDLER_NAME} chars, got {len(name)}"
        )


def validate_task_key(key: str) -> None:
    if not key or len(key) > MAX_TASK_KEY:
        raise ValueError(f"task key must be 1..{MAX_TASK_KEY} chars, got {len(key)}")


def _check_depth(obj: object, depth: int = 0) -> int:
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"metadata nesting exceeds {MAX_METADATA_DEPTH} levels")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check_depth(v, depth + 1)
    return depth


def validate_metadata(metadata: "dict[str, object]") -> None:
    if not metadata:
        return
    try:
        encoded = json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError(
            f"metadata exceeds {MAX_METADATA_BYTES} bytes (got {len(encoded)})"
        )
    _check_depth(metadata)


def validate_commands(count: int) -> None:
    if count > MAX_COMMANDS:
        raise ValueError(f"too many commands: {count} > {MAX_COMMANDS}")


def validate_child_tasks(count: int) -> None:
    if count > MAX_CHILD_TASKS:
        raise ValueError(f"too many child tasks: {count} > {MAX_CHILD_TASKS}")


def validate_output_payload(size: int) -> None:
    if size > MAX_OUTPUT_PAYLOAD_BYTES:
        raise ValueError(
            f"output artifact exceeds {MAX_OUTPUT_PAYLOAD_BYTES} bytes (got {size})"
        )


def validate_create_task(handler: str, key: str, metadata: object) -> None:
    """The per-command input checks every backend must apply when a handler
    creates a child task: handler-name / task-key length and metadata depth+size.
    Centralized so the file and sqlalchemy stores cannot drift."""
    validate_handler_name(handler)
    validate_task_key(key)
    if isinstance(metadata, dict):
        validate_metadata(metadata)


def validate_task_policies(retry: object, side_effect: object) -> None:
    """A NON_IDEMPOTENT task must never be auto-retried, so its retry policy
    must cap attempts at 1. Caught at creation time so an unsafe combination
    fails fast instead of relying solely on the runtime's non-idempotent retry
    guard."""
    from .models import SideEffectMode

    mode = getattr(getattr(side_effect, "mode", None), "value", None)
    max_attempts = getattr(retry, "max_attempts", 1)
    if mode == SideEffectMode.NON_IDEMPOTENT.value and max_attempts > 1:
        raise ValueError(
            "NON_IDEMPOTENT tasks must have max_attempts=1 "
            f"(got {max_attempts})"
        )


def validate_job_budget(budget: object) -> None:
    """A job always owns at least its root task, so ``max_tasks`` -- when set --
    must be >= 1 or the root task can never coexist with its own budget. The
    other caps (depth/attempts/runtime/tokens/cost) are only bounded below by
    positivity when set. ``max_tasks`` is the one that can brick a legal job."""
    max_tasks = getattr(budget, "max_tasks", None)
    if max_tasks is not None and max_tasks < 1:
        raise ValueError(
            "job budget max_tasks must be >= 1 because every job has a root task"
        )
    max_depth = getattr(budget, "max_depth", None)
    if max_depth is not None and max_depth < 0:
        raise ValueError("job budget max_depth must be >= 0")
    max_attempts = getattr(budget, "max_attempts", None)
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("job budget max_attempts must be >= 1")
    max_runtime = getattr(budget, "max_runtime_seconds", None)
    if max_runtime is not None and max_runtime <= 0:
        raise ValueError("job budget max_runtime_seconds must be > 0")


__all__: "list[str]" = [
    "validate_handler_name",
    "validate_task_key",
    "validate_metadata",
    "validate_commands",
    "validate_child_tasks",
    "validate_output_payload",
    "validate_create_task",
    "validate_task_policies",
    "validate_job_budget",
    "MAX_OUTPUT_PAYLOAD_BYTES",
]
