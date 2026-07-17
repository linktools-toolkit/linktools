#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunSnapshot: an immutable, replayable capture of a single run."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..artifact.models import ResourceSnapshotRef


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: str
    run_record_artifact_id: str
    run_definition_artifact_id: str
    input_artifact_id: str
    output_artifact_id: "str | None" = None
    event_artifact_ids: "tuple[str, ...]" = ()
    resource_snapshots: "tuple[ResourceSnapshotRef, ...]" = ()
    task_attempt_id: "str | None" = None
    model_usage: "Mapping[str, object]" = field(default_factory=dict)
    metadata: "Mapping[str, object]" = field(default_factory=dict)


__all__: "list[str]" = ["RunSnapshot"]
