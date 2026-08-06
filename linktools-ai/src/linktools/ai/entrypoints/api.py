#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Production API composition root."""

from dataclasses import dataclass

from ..ports.runtime import Runtime
from ..ports.workflow import WorkflowGateway


@dataclass(frozen=True, slots=True)
class ApiComposition:
    """Explicit dependencies exposed to the HTTP adapter."""

    runtime: Runtime
    workflow_gateway: WorkflowGateway
    routes: "tuple[object, ...]" = ()


def build_api(
    runtime: Runtime,
    workflow_gateway: WorkflowGateway,
    routes: "tuple[object, ...]" = (),
) -> ApiComposition:
    """Assemble the API-facing runtime and its workflow boundary."""
    return ApiComposition(runtime, workflow_gateway, routes)


__all__ = ["ApiComposition", "build_api"]
