#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PackageProvider: the CapabilityProvider for ``package`` / ``package-resource``
/ ``package-entrypoint`` tool refs.

- ``package:<id>``            -> prompt catalog only (Level 0), no tools.
- ``package-resource:<id>``   -> Level-1 list/read resource tools for that pkg.
- ``package-entrypoint:<id>`` -> Level-1 list-entrypoints tool (+ opt-in call).

A package NEVER auto-exposes all its resources/entrypoints as tools: only the
explicitly declared package ids become reachable, and only the read/list tools
are added by default."""

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic_ai.toolsets import FunctionToolset

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.ref import CapabilityRef
from ..providers.package import PackageResourceProvider
from ..subagent.runner import SubagentExecutor
from .entrypoint import EntrypointInfo
from .resolver import EntrypointResolver
from .scope import PackageScope
from .toolset import build_package_entrypoint_toolset, build_package_resource_toolset


@dataclass
class PackageProvider:
    """CapabilityProvider for package-scoped capabilities. Depends only on the
    PackageResourceProvider / EntrypointResolver Protocols (the Directory
    implementations are one possible Provider, not a type boundary). The
    entrypoint executor is injected at construction so call_package_entrypoint
    can run scoped agents without runtime mutation."""

    kind: str = "package"
    resource_provider: "PackageResourceProvider | None" = None
    entrypoint_resolver: "EntrypointResolver | None" = None
    entrypoint_executor: "SubagentExecutor | None" = None

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        # ``package:<id>`` (this provider's own kind) -> prompt catalog only.
        if ref.kind == "package":
            return CapabilityBundle(prompt_sections={
                "packages": f"Package declared: {ref.name}. Use package-resource / "
                            f"package-entrypoint tools to inspect it when enabled.",
            })

        scope = PackageScope(package_id=ref.name)
        allowed: "dict[str, PackageScope]" = {ref.name: scope}
        cfg = dict(ref.config)
        cap = context.exposure_policy
        from ..capability.provider import make_event_emitter
        emit = make_event_emitter(context)

        if ref.kind == "package-resource":
            if self.resource_provider is None:
                return CapabilityBundle()
            ts = build_package_resource_toolset(
                self.resource_provider, allowed=allowed,
                max_resources_per_list=cfg.get("max_resources_per_list", cap.max_resources_per_list),
                max_read_bytes=cfg.get("max_read_bytes", cap.max_read_bytes),
                emit=emit,
            )
            return CapabilityBundle(toolsets=(ts,))

        if ref.kind == "package-entrypoint":
            if self.entrypoint_resolver is None:
                return CapabilityBundle()
            allowed_kinds = tuple(cfg.get("allowed_kinds", cap.allowed_entrypoint_kinds))
            allowed_names = cfg.get("allowed_names")
            expose_call = bool(cfg.get("expose_call_tool", False)) and cap.expose_execution_tools
            ts = build_package_entrypoint_toolset(
                self.entrypoint_resolver, allowed=allowed,
                allowed_kinds=allowed_kinds,
                allowed_names=tuple(allowed_names) if allowed_names else None,
                expose_call_tool=expose_call,
                max_entrypoints_per_list=cfg.get(
                    "max_entrypoints_per_package", cap.max_entrypoints_per_package),
                emit=emit,
                executor=self.entrypoint_executor,
                parent_run_id=context.run_id,
                parent_session_id=context.session_id,
            )
            return CapabilityBundle(toolsets=(ts,))

        # An unknown package-* kind slipped through; nothing to expose.
        return CapabilityBundle()
