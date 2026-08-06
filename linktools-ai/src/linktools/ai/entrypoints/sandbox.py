#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sandbox worker composition root."""

from dataclasses import dataclass

from ..ports.runtime import Runtime


@dataclass(frozen=True, slots=True)
class SandboxComposition:
    """Worker inputs for the separately released sandbox profile."""

    runtime: Runtime
    worker: object
    workflows: "tuple[object, ...]"
    activities: "tuple[object, ...]"

    def register(self) -> None:
        """Register fixed sandbox workflows and Activities on the Worker."""
        self.worker.register_workflows(self.workflows)
        self.worker.register_activities(self.activities)


def build_sandbox_worker(
    runtime: Runtime,
    worker: object,
    workflows: "tuple[object, ...]",
    activities: "tuple[object, ...]",
) -> SandboxComposition:
    """Assemble the sandbox Worker without enabling it implicitly."""
    return SandboxComposition(runtime, worker, workflows, activities)


__all__ = ["SandboxComposition", "build_sandbox_worker"]
