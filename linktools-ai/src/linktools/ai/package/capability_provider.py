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
from typing import ClassVar


from ..capability.models import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.models import CapabilityRef
from ..providers.package import PackageResourceProvider
from ..run.identity import ParentRunIdentity
from ..tool.models import ToolDescriptor
from ..tool.models import ToolContribution, declared_tool_definitions
from ..subagent.runner import SubagentExecutor
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
    # Declares every kind this one provider handles, so the runtime registers
    # it once under all three instead of alias-registering three copies.
    supported_kinds: "ClassVar[tuple[str, ...]]" = (
        "package", "package-resource", "package-entrypoint"
    )
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
            return CapabilityBundle(
                prompt_sections={
                    "packages": f"Package declared: {ref.name}. Use package-resource / "
                    f"package-entrypoint tools to inspect it when enabled.",
                }
            )

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
                self.resource_provider,
                allowed=allowed,
                max_resources_per_list=cfg.get(
                    "max_resources_per_list", cap.max_resources_per_list
                ),
                max_read_bytes=cfg.get("max_read_bytes", cap.max_read_bytes),
                emit=emit,
            )
            pkw = dict(
                source="package",
                capability_kind="package-resource",
                capability_name=ref.name,
                category="package-read",
                risk="low",
                mutating=False,
            )
            descriptors = (
                ToolDescriptor(name="list_package_resources", **pkw),
                ToolDescriptor(name="read_package_resource", **pkw),
            )
            contrib = ToolContribution(tools=declared_tool_definitions(ts, descriptors))
            return CapabilityBundle(tool_contributions=(contrib,))

        if ref.kind == "package-entrypoint":
            if self.entrypoint_resolver is None:
                return CapabilityBundle()
            allowed_kinds = tuple(
                cfg.get("allowed_kinds", cap.allowed_entrypoint_kinds)
            )
            allowed_names = cfg.get("allowed_names")
            expose_call = (
                bool(cfg.get("expose_call_tool", False)) and cap.expose_execution_tools
            )
            # Same ParentRunIdentity shape every spawner builds -- root_run_id
            # comes from context.root_run_id (the ACTUAL root of the chain),
            # not context.run_id, so a package entrypoint nested under an
            # existing subagent chain doesn't truncate lineage to itself.
            parent = None
            if context.run_id is not None and context.session_id is not None:
                parent = ParentRunIdentity(
                    run_id=context.run_id,
                    root_run_id=context.root_run_id or context.run_id,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    tenant_id=context.tenant_id,
                    workspace=context.workspace,
                )
            ts = build_package_entrypoint_toolset(
                self.entrypoint_resolver,
                allowed=allowed,
                allowed_kinds=allowed_kinds,
                allowed_names=tuple(allowed_names) if allowed_names else None,
                expose_call_tool=expose_call,
                max_entrypoints_per_list=cfg.get(
                    "max_entrypoints_per_package", cap.max_entrypoints_per_package
                ),
                emit=emit,
                executor=self.entrypoint_executor,
                parent=parent,
            )
            ekw = dict(
                source="package",
                capability_kind="package-entrypoint",
                capability_name=ref.name,
            )
            descs = [
                ToolDescriptor(
                    name="list_package_entrypoints",
                    category="discovery",
                    risk="low",
                    mutating=False,
                    **ekw,
                )
            ]
            if expose_call:
                descs.append(
                    ToolDescriptor(
                        name="call_package_entrypoint",
                        category="package-execute",
                        risk="high",
                        mutating=True,
                        **ekw,
                    )
                )
            contrib = ToolContribution(
                tools=declared_tool_definitions(ts, tuple(descs))
            )
            return CapabilityBundle(tool_contributions=(contrib,))

        # An unknown package-* kind slipped through; nothing to expose.
        return CapabilityBundle()
