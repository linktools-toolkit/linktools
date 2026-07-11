#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityAssembler: turns an AgentSpec.tools declaration into one merged
CapabilityBundle by dispatching each tool ref to its CapabilityProvider.

Resolution rules:
  - unknown kind (no provider registered)        -> CapabilityResolutionError
  - a provider rejects its ref (not found, ...)  -> propagates that error
  - two capabilities produce the same tool name  -> CapabilityConflictError
    (never silently overwritten)
  - per-capability / total tool counts exceeding
    CapabilityToolExposurePolicy caps            -> CapabilityConflictError
  - failures carry agent_id + ref for diagnostics

A bare ToolRef with kind None (legacy ``tools: [file]``) is treated as
``builtin`` so existing specs resolve unchanged."""

from typing import TYPE_CHECKING, Any, Mapping

from ..agent.spec import ToolRef
from ..errors import CapabilityConflictError, CapabilityResolutionError
from ..events.payloads import (
    CapabilityResolveCompleted,
    CapabilityResolveStarted,
    ToolExposureDenied,
)
from .bundle import CapabilityBundle
from .exposure import is_descriptor_exposable
from .provider import CapabilityContext, CapabilityProvider
from .ref import CapabilityRef

if TYPE_CHECKING:
    from ..agent.spec import AgentSpec


async def _emit(context: CapabilityContext, payload) -> None:
    """Append a capability-lifecycle event when an EventStore is wired on the
    context; a no-op otherwise so resolution stays side-effect-free by default."""
    store = context.event_store
    run_id = context.run_id
    if store is None or run_id is None:
        return
    await store.append(
        stream_id=run_id, run_id=run_id,
        root_run_id=context.root_run_id or run_id, parent_run_id=context.parent_run_id,
        session_id=context.session_id or run_id, runnable_id=context.agent_id,
        payload=payload,
    )

class CapabilityAssembler:
    """Holds a kind -> CapabilityProvider map and resolves an AgentSpec's tools
    into a single merged bundle. What counts as a valid capability kind is
    entirely determined by which providers are registered -- there is no
    separate hardcoded allowlist to keep in sync with the actual provider set."""

    def __init__(self, providers: "Mapping[str, CapabilityProvider]") -> None:
        self._providers: "dict[str, CapabilityProvider]" = dict(providers)

    @property
    def providers(self) -> "Mapping[str, CapabilityProvider]":
        return dict(self._providers)

    def register(self, provider: CapabilityProvider) -> None:
        """Register a provider for every kind it supports. Raises
        CapabilityConflictError if ANY of its kinds is already registered --
        silently overwriting a wired provider is never the right default. Call
        replace() to override intentionally. A provider with multiple
        supported_kinds (e.g. PackageProvider) is registered under all of them
        from this one call."""
        from .provider import provider_kinds
        kinds = provider_kinds(provider)
        for k in kinds:
            if k in self._providers:
                raise CapabilityConflictError(
                    f"capability provider already registered for kind {k!r}; "
                    f"use replace() to intentionally override it"
                )
        for k in kinds:
            self._providers[k] = provider

    def replace(self, provider: CapabilityProvider) -> None:
        """Register a provider for every kind it supports, intentionally
        overriding any provider already registered for those kinds."""
        from .provider import provider_kinds
        for k in provider_kinds(provider):
            self._providers[k] = provider

    async def assemble(
        self,
        spec: "AgentSpec",
        context: CapabilityContext,
    ) -> CapabilityBundle:
        refs = [_to_capability_ref(spec.id, t) for t in (spec.tools or ())]
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
        merged_toolsets: "list[Any]" = []
        merged_contributions: "list[Any]" = []
        merged_pipelines: "list[Any]" = []
        owner_by_tool: "dict[str, str]" = {}
        total_tools = 0
        cap = context.exposure_policy

        for ref in refs:
            provider = self._providers.get(ref.kind)
            if provider is None:
                raise CapabilityResolutionError(
                    f"agent {spec.id}: no capability provider registered for kind "
                    f"{ref.kind!r} (ref {ref})"
                )
            await _emit(context, CapabilityResolveStarted(
                agent_id=spec.id, capability_ref=str(ref)))
            bundle = await provider.resolve(ref, context)

            contributions = list(bundle.tool_contributions)
            if not contributions and bundle.toolsets:
                from ..tool.legacy import LegacyToolsetAdapter
                if all(isinstance(ts, LegacyToolsetAdapter) for ts in bundle.toolsets):
                    contributions = [ts.contribution() for ts in bundle.toolsets]
                else:
                    raise CapabilityResolutionError(
                        f"capability {ref} returned raw toolsets without explicit "
                        "ToolContribution; use an explicit legacy adapter"
                    )

            for contrib in contributions:
                if (contrib.toolset is not None and contrib.descriptors
                        and not contrib.legacy_adapter
                        and isinstance(getattr(contrib.toolset, "tools", None), dict)):
                    raise CapabilityResolutionError(
                        f"capability {ref} returned an introspectable raw toolset; "
                        "Provider must return ManagedToolDefinition entries")

            # Assembly-time schema validation: every ManagedToolDefinition that
            # declares a parameters_json_schema must have a well-formed schema.
            # A malformed schema is rejected HERE, not deferred to first call.
            from ..tool.schema_validate import validate_schema
            for c in contributions:
                for md in (c.tools or ()):
                    validate_schema(md.parameters_json_schema)

            # Centralized ToolExposurePolicy gate -- the single place a
            # descriptor's category/mutating flag decides whether it reaches
            # the model, regardless of which Provider produced it. A denied
            # tool is dropped entirely: it appears in neither the merged
            # toolsets nor the merged contributions, so it cannot be called
            # via any path. Both contribution forms are filtered consistently
            # (per-tool ``tools`` and legacy ``toolset+descriptors``); a
            # tools-only contribution (toolset is None) must NOT crash on
            # ``.filtered``.
            exposed_contributions = []
            denied_names: "list[str]" = []
            for contrib in contributions:
                filtered, dropped = filter_contribution(contrib, cap)
                denied_names.extend(dropped)
                if filtered is not None:
                    exposed_contributions.append(filtered)

            if denied_names:
                await _emit(context, ToolExposureDenied(
                    agent_id=spec.id,
                    reason=f"capability {ref}: tools not exposed by policy: {denied_names}"))

            names = tuple(d.name for c in exposed_contributions for d in _contribution_descriptors(c))
            await _emit(context, CapabilityResolveCompleted(
                agent_id=spec.id, capability_ref=str(ref), tool_count=len(names)))
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

            merged_toolsets.extend(c.toolset for c in exposed_contributions if c.toolset is not None)
            merged_contributions.extend(exposed_contributions)
            # Merge pipelines declared by this capability -- a stable Bundle
            # field that must never be silently dropped. The runtime wires the
            # merged pipeline list into the run's SecurityPipeline.
            if bundle.pipelines:
                merged_pipelines.extend(bundle.pipelines)
            # Reject non-empty resources (lifecycle not implemented).
            if bundle.resources:
                raise CapabilityResolutionError(
                    f"agent {spec.id}: capability {ref} returned non-empty resources; "
                    f"resource lifecycle is not implemented in this phase"
                )
            total_tools += len(names)

        if total_tools > cap.max_tools_total:
            await _emit(context, ToolExposureDenied(
                agent_id=spec.id,
                reason=f"{total_tools} tools exceed max_tools_total={cap.max_tools_total}"))
            raise CapabilityConflictError(
                f"agent {spec.id}: {total_tools} tools declared "
                f"(max_tools_total={cap.max_tools_total})"
            )

        return CapabilityBundle(
            prompt_sections=merged_prompt,
            toolsets=tuple(merged_toolsets),
            tool_contributions=tuple(merged_contributions),
            middleware=tuple(),
            resources=tuple(),
            pipelines=tuple(merged_pipelines),
        )


def _to_capability_ref(agent_id: str, tool_ref: ToolRef) -> CapabilityRef:
    kind = tool_ref.kind or "builtin"
    return CapabilityRef(kind=kind, name=tool_ref.name, config=dict(tool_ref.config))


def _contribution_descriptors(contrib) -> "tuple":
    """All descriptors on a contribution -- the per-tool form (preferred) and/or
    the legacy descriptors tuple. The single source for counting, conflict
    detection, and exposure accounting (no toolset introspection)."""
    out: "list" = []
    if contrib.tools:
        out.extend(md.descriptor for md in contrib.tools)
    if contrib.descriptors:
        seen = {d.name for d in out}
        for d in contrib.descriptors:
            if d.name not in seen:
                out.append(d)
    return tuple(out)


def filter_contribution(contrib, policy):
    """Apply ToolExposurePolicy to one ToolContribution.

    Returns ``(filtered_or_None, dropped_names)``. Handles both the per-tool
    ``tools`` form and the legacy ``toolset + descriptors`` form consistently:
    a tools-only contribution (toolset is None) filters its ``tools`` tuple and
    never crashes on ``.filtered``. An empty result returns ``None`` (the
    contribution is dropped entirely) rather than raising.
    """
    descs = _contribution_descriptors(contrib)
    if not descs:
        return None, []
    allowed = tuple(d for d in descs if is_descriptor_exposable(d, policy))
    dropped = [d.name for d in descs if d not in allowed]
    if not allowed:
        return None, dropped
    if len(allowed) == len(descs):
        return contrib, []
    allowed_names = frozenset(d.name for d in allowed)
    from ..tool.contribution import ToolContribution
    if contrib.tools:
        kept = tuple(md for md in contrib.tools if md.descriptor.name in allowed_names)
        return ToolContribution(tools=kept), dropped
    if contrib.toolset is not None:
        kept_descs = tuple(d for d in contrib.descriptors if d.name in allowed_names)
        filtered_toolset = contrib.toolset.filtered(
            lambda _ctx, tool_def, _names=allowed_names: tool_def.name in _names)
        return ToolContribution(toolset=filtered_toolset, descriptors=kept_descs), dropped
    return ToolContribution(descriptors=allowed), dropped


# Both names refer to the same orchestrator.
CapabilityResolver = CapabilityAssembler
