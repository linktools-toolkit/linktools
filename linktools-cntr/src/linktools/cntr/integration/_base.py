#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata and consumer lifecycle contract."""
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from typing import AbstractSet, Mapping, Optional
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext


class Integration:
    """A consumer-specific declaration with an optional producer-local ID."""

    consumer: str = ""
    local_id: "Optional[str]" = None
    requires_local_id: bool = False


Integrations = Iterable[Integration]


class IntegrationConsumer:
    """Consumer policy invoked by the shared Compose orchestration path."""

    generated = False
    application_order = 0
    uses_generation_label = True
    plan_warnings: "tuple[str, ...]" = ()

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    def get_runtime_requirements(self, required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
        """Return installed providers and the services required from each."""
        return {}

    def on_applied(self, context: "EventContext", service: str) -> None:
        pass

    def needs_apply(self, candidate: "GeneratedCandidate", context: "EventContext") -> bool:
        return candidate.changed

    def needs_bootstrap(self, services: "Iterable[str]", running_services: "AbstractSet[str]") -> bool:
        return False

    def on_bootstrap(self, context: "EventContext") -> str:
        """Return the confirmed bootstrap generation for rollback."""
        raise NotImplementedError

    def on_prepare(self, context: "EventContext") -> None:
        pass

    def on_render(self, generation_id: str) -> "dict[str, str]":
        raise NotImplementedError

    def on_validate(self, context: "EventContext", candidate: "GeneratedCandidate") -> None:
        raise NotImplementedError

    def on_apply(self, context: "EventContext", candidate: "GeneratedCandidate",
                 services: "Iterable[str]") -> None:
        raise NotImplementedError
