#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared declaration metadata and consumer lifecycle contract."""
import re
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from typing import AbstractSet, Mapping, Optional
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext
    from ..manager import ContainerManager
    from ..runtime.structured import CommandResult


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

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    @classmethod
    def runtime_requirements(cls, manager: "ContainerManager",
                             required: "AbstractSet[str]") -> "Mapping[str, Iterable[str]]":
        """Return installed providers and the services required from each."""
        return {}

    @classmethod
    def plan_warnings(cls) -> "tuple[str, ...]":
        return ()

    @classmethod
    def validation_failed(cls, result: "CommandResult") -> bool:
        return not result.succeeded

    @classmethod
    def validation_diagnostic(cls, manager: "ContainerManager", result: "CommandResult") -> str:
        """Keep the source location without exposing expanded configuration."""
        match = re.search(r" in ([/A-Za-z0-9_.-]+):(\d+)", result.stderr)
        return " at {}:{}".format(*match.groups()) if match else ""

    @classmethod
    def after_apply(cls, manager: "ContainerManager", context: "EventContext", service: str) -> None:
        pass

    def needs_apply(self, candidate: "GeneratedCandidate", context: "EventContext") -> bool:
        return candidate.changed

    def needs_bootstrap(self, services: "Iterable[str]", running_services: "AbstractSet[str]") -> bool:
        return False

    def bootstrap(self, context: "EventContext") -> str:
        """Return the confirmed bootstrap generation for rollback."""
        raise NotImplementedError

    def prepare(self, context: "EventContext") -> None:
        raise NotImplementedError

    def render(self, generation_id: str) -> "dict[str, str]":
        raise NotImplementedError

    def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        raise NotImplementedError

    def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
              services: "Iterable[str]") -> None:
        raise NotImplementedError
