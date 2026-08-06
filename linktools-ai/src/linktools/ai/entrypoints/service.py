#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Production service composition root."""

from dataclasses import dataclass

from ..ports.runtime import Runtime


@dataclass(frozen=True, slots=True)
class ServiceComposition:
    """Worker registration inputs assembled by the service entry point."""

    runtime: Runtime
    workflows: "tuple[object, ...]"
    activities: "tuple[object, ...]"

    def register(self, worker: object) -> None:
        """Register the fixed workflow and Activity sets on a Worker."""
        worker.register_workflows(self.workflows)
        worker.register_activities(self.activities)


def build_service(
    runtime: Runtime,
    workflows: "tuple[object, ...]",
    activities: "tuple[object, ...]",
) -> ServiceComposition:
    """Assemble the ordinary Temporal Worker composition."""
    return ServiceComposition(runtime, workflows, activities)


__all__ = ["ServiceComposition", "build_service"]
