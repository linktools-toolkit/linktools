#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A SecurityEventEmitter that records sanitized events in memory instead of
persisting them. Used by Runtime.inspect so a capability resolution that emits
SecurityDegraded (e.g. MCP best-effort discovery) behaves the same under
inspection as under a real run -- the degradation is observable, not silently
swallowed or turned into a hard failure.

Nothing is written to an EventStore: inspection has no run_id and must not
produce audit side effects. The sanitizer still runs so any secret in an event
field is redacted before the event is exposed via the warnings API."""

from typing import Any

from .emitter import SecurityEventSanitizer


class CollectingSecurityEventEmitter:
    def __init__(self, *, sanitizer: "SecurityEventSanitizer | Any") -> None:
        self._sanitizer = sanitizer
        self._security_events: "list[Any]" = []
        self._observability_events: "list[Any]" = []

    async def emit_security(self, event: Any) -> None:
        self._security_events.append(self._sanitizer.sanitize(event))

    async def emit_observability(self, event: Any) -> None:
        self._observability_events.append(self._sanitizer.sanitize(event))

    @property
    def security_events(self) -> "tuple[Any, ...]":
        return tuple(self._security_events)

    @property
    def observability_events(self) -> "tuple[Any, ...]":
        return tuple(self._observability_events)
