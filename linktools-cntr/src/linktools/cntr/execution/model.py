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
    phase: str
    args: "tuple[str, ...]"
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
