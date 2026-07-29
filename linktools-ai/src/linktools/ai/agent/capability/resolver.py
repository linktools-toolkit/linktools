#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityResolver: turns an AgentSpec.tools declaration into one merged
CapabilityBundle by dispatching each tool ref to its CapabilityProvider.

Resolution rules:
  - unknown kind (no provider registered)        -> CapabilityResolutionError
  - a provider rejects its ref (not found, ...)  -> propagates that error
  - two capabilities produce the same tool name  -> CapabilityConflictError
    (never silently overwritten)
  - per-capability / total tool counts exceeding
    CapabilityToolExposurePolicy caps            -> CapabilityConflictError
  - failures carry agent_id + ref for diagnostics

"""

from typing import TYPE_CHECKING, Any, Mapping

from ...errors import CapabilityConflictError, CapabilityResolutionError
from ...observability.events.payloads import (
    CapabilityResolveCompleted,
    CapabilityResolveStarted,
    ToolExposureDenied,
)
from .models import CapabilityBundle
from .exposure import is_descriptor_exposable
from .provider import CapabilityContext, CapabilityProvider
from .registry import CapabilityProviderRegistry

if TYPE_CHECKING:
    from ..spec import AgentSpec


async def _emit(context: CapabilityContext, payload) -> None:
    """Emit a capability-lifecycle event through the context's
    SecurityEventEmitter when one is wired; a no-op otherwise so resolution
    stays side-effect-free by default. Never a direct event store reference.

    Routing follows the payload's criticality: a SECURITY_CRITICAL event
    (e.g. ToolExposureDenied -- an audit record of an exposure-policy denial)
    goes through emit_security so a persistence failure blocks (fail-closed,
    preserving the audit trail); OBSERVABILITY lifecycle markers go through
    emit_observability (best-effort)."""
    emitter = context.security_event_emitter
    if emitter is None:
        return
    from ...observability.events.models import EventCriticality

    if getattr(type(payload), "criticality", None) is EventCriticality.SECURITY_CRITICAL:
        await emitter.emit_security(payload)
    else:
        await emitter.emit_observability(payload)


class CapabilityResolver:
    """Resolves an AgentSpec's tool refs into one merged CapabilityBundle by
    dispatching each ref to its CapabilityProvider via the runtime
    :class:`CapabilityProviderRegistry`.

    The provider store (kind -> provider, plus register/replace) lives on the
    registry -- the single runtime registry. The resolver holds a
    registry reference and owns per-spec resolution only. Constructed from either
    an existing registry or a raw kind -> provider mapping (convenience: the
    mapping is wrapped in a registry). What counts as a valid capability kind is
    entirely determined by which providers the registry holds -- there is no
    separate hardcoded allowlist to keep in sync with the provider set."""

    def __init__(
        self, providers: "CapabilityProviderRegistry | Mapping[str, CapabilityProvider]"
    ) -> None:
        if isinstance(providers, CapabilityProviderRegistry):
            self._registry = providers
        else:
            self._registry = CapabilityProviderRegistry(providers)

    @property
    def registry(self) -> CapabilityProviderRegistry:
        return self._registry

    @property
    def providers(self) -> "Mapping[str, CapabilityProvider]":
        # Read-only view over the registry's store (mutation goes through the
        # registry's register/replace).
        return self._registry.providers

    async def resolve(
        self,
        spec: "AgentSpec",
        context: CapabilityContext,
    ) -> CapabilityBundle:
        refs = list(spec.tools or ())
        # detect duplicate identical refs early -- a spec listing the same
        # capability twice is almost always a mistake.
        seen_refs: "set[tuple[str, str]]" = set()
        for ref in refs:
            key = (ref.kind, ref.name)
            if key in seen_refs:
                raise CapabilityConflictError(
                    f"agent {spec.id}: duplicate tool declaration {ref}"
                )
            seen_refs.add(key)

        merged_prompt: "dict[str, str]" = {}
        merged_contributions: "list[Any]" = []
        owner_by_tool: "dict[str, str]" = {}
        total_tools = 0
        cap = context.exposure_policy

        for ref in refs:
            provider = self._registry.get(ref.kind)
            if provider is None:
                raise CapabilityResolutionError(
                    f"agent {spec.id}: no capability provider registered for kind "
                    f"{ref.kind!r} (ref {ref})"
                )
            await _emit(
                context,
                CapabilityResolveStarted(agent_id=spec.id, capability_ref=str(ref)),
            )
            bundle = await provider.resolve(ref, context)

            contributions = list(bundle.tool_contributions)

            # Assembly-time schema validation: every ManagedToolDefinition that
            # declares a parameters_json_schema must have a well-formed schema.
            # A malformed schema is rejected HERE, not postponed to first call.
            from ...tool.schema import validate_schema

            for c in contributions:
                for md in c.tools or ():
                    validate_schema(md.parameters_json_schema)

            # Centralized ToolExposurePolicy gate -- the single place a
            # descriptor's category/mutating flag decides whether it reaches
            # the model, regardless of which Provider produced it. A denied
            # tool is dropped entirely: it appears in neither the merged
            # toolsets nor the merged contributions, so it cannot be called
            # via any path. Both contribution forms are filtered consistently
            # A tools-only contribution must NOT crash on filtering.
            exposed_contributions = []
            denied_names: "list[str]" = []
            for contrib in contributions:
                filtered, dropped = filter_contribution(contrib, cap)
                denied_names.extend(dropped)
                if filtered is not None:
                    exposed_contributions.append(filtered)

            if denied_names:
                await _emit(
                    context,
                    ToolExposureDenied(
                        agent_id=spec.id,
                        reason=f"capability {ref}: tools not exposed by policy: {denied_names}",
                    ),
                )

            names = tuple(
                d.name
                for c in exposed_contributions
                for d in contribution_descriptors(c)
            )
            await _emit(
                context,
                CapabilityResolveCompleted(
                    agent_id=spec.id, capability_ref=str(ref), tool_count=len(names)
                ),
            )
            if len(names) > cap.max_tools_per_capability:
                raise CapabilityConflictError(
                    f"agent {spec.id}: capability {ref} exposes {len(names)} tools "
                    f"(max_tools_per_capability={cap.max_tools_per_capability})"
                )
            for n in names:
                if n in owner_by_tool:
                    raise CapabilityConflictError(
                        f"agent {spec.id}: tool {n!r} produced by both "
                        f"{owner_by_tool[n]} and {ref}"
                    )
                owner_by_tool[n] = str(ref)

            for section, text in bundle.prompt_sections.items():
                if section in merged_prompt and merged_prompt[section]:
                    merged_prompt[section] = merged_prompt[section] + "\n" + text
                else:
                    merged_prompt[section] = text

            merged_contributions.extend(exposed_contributions)
            total_tools += len(names)

        if total_tools > cap.max_tools_total:
            await _emit(
                context,
                ToolExposureDenied(
                    agent_id=spec.id,
                    reason=f"{total_tools} tools exceed max_tools_total={cap.max_tools_total}",
                ),
            )
            raise CapabilityConflictError(
                f"agent {spec.id}: {total_tools} tools declared "
                f"(max_tools_total={cap.max_tools_total})"
            )

        return CapabilityBundle(
            prompt_sections=merged_prompt,
            tool_contributions=tuple(merged_contributions),
        )


def contribution_descriptors(contrib) -> "tuple":
    """The descriptors on a contribution -- one per ManagedToolDefinition in
    ``tools``. The single source for counting, conflict detection, and exposure
    accounting (no toolset introspection)."""
    return tuple(md.descriptor for md in contrib.tools)


def filter_contribution(contrib, policy):
    """Apply ToolExposurePolicy to one ToolContribution.

    Returns ``(filtered_or_None, dropped_names)``. Filters the ``tools`` tuple
    by descriptor; an empty result returns ``None`` (the contribution is dropped
    entirely) rather than raising.
    """
    descs = contribution_descriptors(contrib)
    if not descs:
        return None, []
    allowed = tuple(d for d in descs if is_descriptor_exposable(d, policy))
    dropped = [d.name for d in descs if d not in allowed]
    if not allowed:
        return None, dropped
    if len(allowed) == len(descs):
        return contrib, []
    allowed_names = frozenset(d.name for d in allowed)
    from ...tool.models import ToolContribution

    kept = tuple(md for md in contrib.tools if md.descriptor.name in allowed_names)
    return ToolContribution(tools=kept), dropped
