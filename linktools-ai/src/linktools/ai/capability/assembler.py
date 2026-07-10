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
from ..errors import CapabilityConflictError, CapabilityResolutionError, InvalidSpecError
from ..events.payloads import (
    CapabilityResolveCompleted,
    CapabilityResolveStarted,
    ToolExposureDenied,
)
from .bundle import CapabilityBundle
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

# Recognized capability kinds. A kind outside this set is a
# structurally invalid tool ref (InvalidSpecError); a known kind with no
# registered provider is a wiring gap (CapabilityResolutionError).
KNOWN_CAPABILITY_KINDS = frozenset({
    "builtin", "skill", "mcp", "subagent",
    "package", "package-resource", "package-entrypoint",
})


class CapabilityAssembler:
    """Holds a kind -> CapabilityProvider map and resolves an AgentSpec's tools
    into a single merged bundle."""

    def __init__(self, providers: "Mapping[str, CapabilityProvider]") -> None:
        self._providers: "dict[str, CapabilityProvider]" = dict(providers)

    @property
    def providers(self) -> "Mapping[str, CapabilityProvider]":
        return dict(self._providers)

    def register(self, provider: CapabilityProvider) -> None:
        self._providers[provider.kind] = provider

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
        owner_by_tool: "dict[str, str]" = {}
        total_tools = 0
        cap = context.exposure_policy

        for ref in refs:
            if ref.kind not in KNOWN_CAPABILITY_KINDS:
                raise InvalidSpecError(
                    f"agent {spec.id}: unknown capability kind {ref.kind!r} (ref {ref})"
                )
            provider = self._providers.get(ref.kind)
            if provider is None:
                raise CapabilityResolutionError(
                    f"agent {spec.id}: no capability provider registered for kind "
                    f"{ref.kind!r} (ref {ref})"
                )
            await _emit(context, CapabilityResolveStarted(
                agent_id=spec.id, capability_ref=str(ref)))
            bundle = await provider.resolve(ref, context)

            names = toolset_names(bundle.toolsets)
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

            merged_toolsets.extend(bundle.toolsets)
            # Auto-wrap raw toolsets into ToolContributions with conservative
            # descriptors so downstream (ManagedToolAdapter) never needs
            # toolset introspection (compat adaptation).
            if bundle.tool_contributions:
                merged_contributions.extend(bundle.tool_contributions)
            elif bundle.toolsets:
                from ..tool.auto_descriptor import auto_contribute
                for ts in bundle.toolsets:
                    merged_contributions.append(auto_contribute(
                        ts, source=ref.kind, capability_kind=ref.kind,
                        capability_name=ref.name))
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
        )


def _to_capability_ref(agent_id: str, tool_ref: ToolRef) -> CapabilityRef:
    kind = tool_ref.kind or "builtin"
    return CapabilityRef(kind=kind, name=tool_ref.name, config=dict(tool_ref.config))


# Both names refer to the same orchestrator.
CapabilityResolver = CapabilityAssembler
