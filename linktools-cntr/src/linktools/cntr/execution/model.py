#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution Plan model, and the Execution Report recorded during a real
up/restart/down run (opt-in via ``--report``).

``ExecutionPlan`` holds no BaseContainer/Process/callable -- only plain,
JSON-friendly values -- so it can be serialized (``ct-cntr plan --json``)
and unit-tested without any of that machinery. ``ExecutionRecord`` is
stashed on the existing ``EventContext.metadata`` extension point rather
than a new context field, so reporting stays additive. Not printed by
default -- only ``--report`` renders the full list via ``render_report``,
but a failure's phase/container/command (redacted)/duration/error summary
is always logged immediately regardless of ``--report``.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..runtime.structured import redact_command as _redact_command

if TYPE_CHECKING:
    from ..context import EventContext


@dataclass(frozen=True)
class PlannedArtifact:
    path: str
    kind: str
    container: str
    old_sha256: "str | None"
    new_sha256: str
    change: str  # "added" | "changed" | "unchanged"


@dataclass(frozen=True)
class PlannedCommand:
    """``args``/``display_args`` are both already redacted (see
    ``runtime.structured.redact_command``) at construction time -- neither
    field ever holds a raw secret, so serializing the whole plan (including
    via ``dataclasses.asdict``) can never leak one. ``args`` omits ``sudo``,
    for exact comparison against the real runtime argv in tests;
    ``display_args`` is the full, copy-paste-executable command a user
    would actually run, including ``sudo`` when privilege would apply it."""
    phase: str
    args: "tuple[str, ...]"
    display_args: "tuple[str, ...]"
    privilege: bool
    interactive: bool


@dataclass(frozen=True)
class PlannedHook:
    phase: str
    container: "str | None"
    name: str
    opaque: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    action: str
    project: str
    full: bool
    targets: "tuple[str, ...]"
    resolved_containers: "tuple[str, ...]"
    services: "tuple[str, ...]"
    compose_files: "tuple[str, ...]"
    artifacts: "tuple[PlannedArtifact, ...]"
    commands: "tuple[PlannedCommand, ...]"
    hooks: "tuple[PlannedHook, ...]"
    warnings: "tuple[str, ...]"
    preflight: str = "skipped"  # "passed" | "skipped" | "failed"


# ---------------------------------------------------------------------------
# Execution Report
# ---------------------------------------------------------------------------

RECORDS_KEY = "execution_records"


@dataclass(frozen=True)
class ExecutionRecord:
    phase: str
    container: "str | None"
    command: "tuple[str, ...] | None"
    success: bool
    duration: float
    message: "str | None" = None


def get_records(context: "EventContext") -> "list[ExecutionRecord]":
    return context.metadata.setdefault(RECORDS_KEY, [])


def _format_failure(record: "ExecutionRecord") -> str:
    scope = record.container or "(project)"
    command = " ".join(record.command) if record.command else ""
    summary = f"[FAILED] {record.phase} {scope} ({record.duration:.2f}s) {command}".rstrip()
    if record.message:
        summary += f"\n    error: {record.message}"
    return summary


@contextmanager
def record_phase(context: "EventContext", phase: str, command: "tuple[str, ...] | None" = None,
                 container: "str | None" = None, logger=None):
    """Time one phase of a real apply and append an ExecutionRecord,
    whether it succeeds or raises. Re-raises whatever the body raised.

    When ``logger`` is given, a failure is always logged immediately --
    independent of ``--report``: phase/container/command/duration/error
    summary are shown on every failure regardless of whether the full
    report is ever rendered.
    """
    command = _redact_command(command)
    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        record = ExecutionRecord(
            phase=phase, container=container, command=command,
            success=False, duration=time.monotonic() - started, message=str(exc),
        )
        get_records(context).append(record)
        if logger is not None:
            logger.error(_format_failure(record))
        raise
    else:
        get_records(context).append(ExecutionRecord(
            phase=phase, container=container, command=command,
            success=True, duration=time.monotonic() - started,
        ))


def render_report(logger, records: "list[ExecutionRecord]") -> None:
    """Render the full report -- only ever called when --report was
    explicitly requested; a failure's own diagnostic (phase/container/
    command/duration/message) is shown by the caller regardless."""
    for record in records:
        status = "ok" if record.success else "FAILED"
        scope = record.container or "(project)"
        command = " ".join(record.command) if record.command else ""
        logger.info(f"[{status}] {record.phase} {scope} ({record.duration:.2f}s) {command}")
        if not record.success and record.message:
            logger.info(f"    error: {record.message}")
