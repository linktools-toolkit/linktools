#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime capability inspection.

``inspect_capabilities`` is the single assembly-inspection entry point behind
Runtime.inspect. It resolves a spec's capabilities under an in-memory
security-event collector (inspection must not have audit side effects) and
returns an immutable CapabilityInspection that leaks no handlers."""

import dataclasses
from typing import TYPE_CHECKING, Any

from ...agent.spec import AgentSpec
from ...capability.models import CapabilityRuntimeOptions
from ...sandbox.protocols import Sandbox

if TYPE_CHECKING:
    from ...capability.resolver import CapabilityResolver
    from ...capability.models import CapabilityBundle
    from ...capability.models import CapabilityInspection


def _inspection_warnings_from_events(events: "tuple[Any, ...]") -> "tuple[str, ...]":
    """Turn collected SecurityDegraded events into human-readable inspection
    warnings. Only SecurityDegraded is surfaced -- and only its (already
    sanitizer-redacted) reason -- so inspection never leaks a secret or exposes
    the raw event objects."""
    warnings: "list[str]" = []
    from ...events.payloads import SecurityDegraded, TruncatedSecurityEvent

    for event in events:
        if isinstance(event, SecurityDegraded):
            warnings.append(f"security degraded: {event.reason}")
        elif isinstance(event, TruncatedSecurityEvent):
            warnings.append(f"security event truncated: {event.original_event_type}")
    return tuple(warnings)


async def _assemble_internal(
    resolver: "CapabilityResolver | None",
    options: CapabilityRuntimeOptions,
    spec: AgentSpec,
    sandbox: "Sandbox | None",
    security_event_emitter: Any,
) -> "CapabilityBundle":
    """Resolve ``spec.tools`` into a CapabilityBundle under the runtime's
    exposure policy. Returns an empty bundle when no capability providers are
    configured. ``security_event_emitter`` is wired into the CapabilityContext
    so a resolution that degrades can emit its SecurityDegraded event instead
    of failing for want of an emitter."""
    from ...capability.provider import CapabilityContext
    from ...capability.models import CapabilityBundle, requires_capability_resolver

    if resolver is None and requires_capability_resolver(
        tools=spec.tools, sandbox=sandbox
    ):
        from ...errors import RuntimeInitializationError

        raise RuntimeInitializationError(
            "Runtime cannot inspect tools without CapabilityResolver"
        )
    if resolver is None:
        return CapabilityBundle.empty()
    context = CapabilityContext(
        agent_id=spec.id,
        exposure_policy=options.tool_exposure,
        sandbox=sandbox,
        security_event_emitter=security_event_emitter,
    )
    return await resolver.assemble(spec, context)


async def inspect_capabilities(
    *,
    resolver: "CapabilityResolver | None",
    options: CapabilityRuntimeOptions,
    spec: AgentSpec,
    sandbox: "Sandbox | None",
) -> "CapabilityInspection":
    """A stable, immutable view of what ``spec`` resolves to: exposed tool
    descriptors, merged prompt sections, and any warnings. Leaks no mutable
    internal state (no handlers). A capability that degrades during resolution
    emits a SecurityDegraded event into an in-memory collector rather than an
    EventStore; those events are surfaced as warnings so inspection reflects
    the same degradation a real run would observe."""
    from ...capability.models import CapabilityInspection
    from ...governance.security.emitter import CollectingSecurityEventEmitter
    from ...governance.security.emitter import DefaultSecurityEventSanitizer

    collector = CollectingSecurityEventEmitter(
        sanitizer=DefaultSecurityEventSanitizer()
    )
    bundle = await _assemble_internal(
        resolver, options, spec, sandbox, security_event_emitter=collector
    )
    inspection = CapabilityInspection.from_bundle(
        bundle, exposure_policy=options.tool_exposure
    )
    return dataclasses.replace(
        inspection,
        warnings=(
            *inspection.warnings,
            *_inspection_warnings_from_events(collector.security_events),
        ),
    )
