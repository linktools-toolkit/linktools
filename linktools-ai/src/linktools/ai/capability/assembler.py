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
from .provider import CapabilityContext, CapabilityProvider, toolset_names
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
        root_run_id=context.root_run_id or run_id, parent_run_id=None,
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

            # Auto-wrap raw toolsets into ToolContributions with conservative
            # descriptors so downstream (ManagedToolAdapter) never needs
            # toolset introspection. This is a compat adaptation: every
            # first-party Provider already returns explicit descriptors, so a
            # Provider hitting this path hasn't migrated yet. Surface it via
            # DeprecationWarning (compat behavior must be observable) rather
            # than silently introspecting.
            contributions = list(bundle.tool_contributions)
            if not contributions and bundle.toolsets:
                import warnings
                warnings.warn(
                    f"capability {ref} returned raw toolsets without ToolContribution "
                    f"descriptors; auto-introspection is deprecated -- return "
                    f"ToolContribution with explicit ToolDescriptors",
                    DeprecationWarning,
                    stacklevel=2,
                )
                from ..tool.auto_descriptor import auto_contribute
                contributions = [
                    auto_contribute(ts, source=ref.kind, capability_kind=ref.kind,
                                     capability_name=ref.name)
                    for ts in bundle.toolsets
                ]

            # Descriptor/handler resolution: for the toolset+descriptors form
            # with an introspectable toolset, every declared descriptor must
            # resolve to a unique raw handler in the toolset -- otherwise the
            # descriptor would only fail-closed at call time. Catch the mismatch
            # here, at assembly. Opaque toolsets (no .tools dict, e.g. MCP's
            # FilteredToolset) are exempt: their handlers are the forwarding
            # closures, not introspectable functions.
            from ..tool.auto_descriptor import ToolsetAdapter, _toolset_tool_names
            for idx, contrib in enumerate(contributions):
                if contrib.tools:
                    continue  # per-tool form: handler is explicit per definition
                if not contrib.toolset or not contrib.descriptors:
                    continue
                names_in_toolset = set(_toolset_tool_names(contrib.toolset))
                if not names_in_toolset:
                    continue  # opaque toolset -- skip (forwarding model)
                missing = [d.name for d in contrib.descriptors
                           if d.name not in names_in_toolset]
                if missing:
                    raise CapabilityResolutionError(
                        f"agent {spec.id}: capability {ref} declared descriptors "
                        f"with no matching handler in the toolset: {missing}")
                # Populate the per-tool ManagedToolDefinition form from the
                # introspectable toolset so downstream (the runner) governs each
                # tool via its own explicit descriptor+handler pair -- the
                # preferred per-tool model, derived here for any toolset whose
                # handlers are extractable. Opaque toolsets (MCP) keep the
                # toolset+descriptors form.
                from ..tool.contribution import ManagedToolDefinition, ToolContribution
                built = tuple(
                    ManagedToolDefinition(
                        descriptor=d,
                        handler=ToolsetAdapter.extract_handler(contrib.toolset, d.name))
                    for d in contrib.descriptors
                )
                contributions[idx] = ToolContribution(
                    toolset=contrib.toolset, descriptors=contrib.descriptors, tools=built)

            # Centralized ToolExposurePolicy gate -- the single place a
            # descriptor's category/mutating flag decides whether it reaches
            # the model, regardless of which Provider produced it. A denied
            # tool is dropped entirely: it appears in neither the merged
            # toolsets nor the merged contributions, so it cannot be called
            # via any path (managed or raw).
            exposed_contributions = []
            denied_names: "list[str]" = []
            for contrib in contributions:
                exposed = tuple(
                    d for d in contrib.descriptors if is_descriptor_exposable(d, cap))
                denied_names.extend(
                    d.name for d in contrib.descriptors if d not in exposed)
                if not contrib.descriptors:
                    # No descriptors at all (shouldn't happen post auto-wrap,
                    # but never silently expose an unclassified toolset).
                    continue
                if not exposed:
                    continue
                if len(exposed) == len(contrib.descriptors):
                    exposed_contributions.append(contrib)
                else:
                    allowed_names = {d.name for d in exposed}
                    from ..tool.contribution import ToolContribution
                    filtered_toolset = contrib.toolset.filtered(
                        lambda _ctx, tool_def, _names=allowed_names: tool_def.name in _names
                    )
                    exposed_contributions.append(
                        ToolContribution(toolset=filtered_toolset, descriptors=exposed))

            if denied_names:
                await _emit(context, ToolExposureDenied(
                    agent_id=spec.id,
                    reason=f"capability {ref}: tools not exposed by policy: {denied_names}"))

            names = tuple(d.name for c in exposed_contributions for d in c.descriptors)
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

            merged_toolsets.extend(c.toolset for c in exposed_contributions)
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


# Both names refer to the same orchestrator.
CapabilityResolver = CapabilityAssembler
