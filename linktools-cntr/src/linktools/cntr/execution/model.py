#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution Plan model (Spec Part V section 33).

``ExecutionPlan`` holds no BaseContainer/Process/callable -- only plain,
JSON-friendly values -- so it can be serialized (``ct-cntr plan --json``)
and unit-tested without any of that machinery.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
