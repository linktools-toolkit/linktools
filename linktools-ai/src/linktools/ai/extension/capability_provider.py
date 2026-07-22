#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider: the CapabilityProvider for ``extension`` / ``extension-asset``
/ ``extension-entrypoint`` tool refs.

- ``extension:<id>``            -> prompt catalog only (Level 0), no tools.
- ``extension-asset:<id>``   -> Level-1 list/read asset tools for that extension.
- ``extension-entrypoint:<id>`` -> Level-1 list-entrypoints tool (+ opt-in call).

An extension NEVER auto-exposes all its assets/entrypoints as tools: only the
explicitly declared extension ids become reachable, and only the read/list tools
are added by default."""

from dataclasses import dataclass
from typing import ClassVar


from ..capability.models import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.models import CapabilityRef
from .spec import ExtensionContentSource
from ..run.identity import ParentRunIdentity
from ..tool.models import ToolDescriptor
from ..tool.models import ToolContribution, declared_tool_definitions
from ..subagent.runner import SubagentExecutorProtocol
from .resolver import EntrypointResolver
from .scope import ExtensionScope
from .toolset import build_extension_entrypoint_toolset, build_extension_resource_toolset


@dataclass
class ExtensionProvider:
    """CapabilityProvider for extension-scoped capabilities. Depends only on the
    ExtensionContentSource / EntrypointResolver Protocols (the Directory
    implementations are one possible Provider, not a type boundary). The
    entrypoint executor is injected at construction so call_extension_entrypoint
    can run scoped agents without runtime mutation."""

    kind: str = "extension"
    # Declares every kind this one provider handles, so the runtime registers
    # it once under all three instead of alias-registering three copies.
    supported_kinds: "ClassVar[tuple[str, ...]]" = (
        "extension",
        "extension-asset",
        "extension-entrypoint",
    )
    content_source: "ExtensionContentSource | None" = None
    entrypoint_resolver: "EntrypointResolver | None" = None
    entrypoint_executor: "SubagentExecutorProtocol | None" = None

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        # ``extension:<id>`` (this provider's own kind) -> prompt catalog only.
        if ref.kind == "extension":
            return CapabilityBundle(
                prompt_sections={
                    "extensions": f"Extension declared: {ref.name}. Use extension-asset / "
                    f"extension-entrypoint tools to inspect it when enabled.",
                }
            )

        scope = ExtensionScope(extension_id=ref.name)
        allowed: "dict[str, ExtensionScope]" = {ref.name: scope}
        cfg = dict(ref.config)
        cap = context.exposure_policy
        from ..capability.provider import make_event_emitter

        emit = make_event_emitter(context)

        if ref.kind == "extension-asset":
            if self.content_source is None:
                return CapabilityBundle()
            ts = build_extension_resource_toolset(
                self.content_source,
                allowed=allowed,
                max_resources_per_list=cfg.get(
                    "max_resources_per_list", cap.max_resources_per_list
                ),
                max_read_bytes=cfg.get("max_read_bytes", cap.max_read_bytes),
                emit=emit,
            )
            pkw = dict(
                source="extension",
                capability_kind="extension-asset",
                capability_name=ref.name,
                category="extension-read",
                risk="low",
                mutating=False,
            )
            descriptors = (
                ToolDescriptor(name="list_extension_content", **pkw),
                ToolDescriptor(name="read_extension_content", **pkw),
            )
            contrib = ToolContribution(tools=declared_tool_definitions(ts, descriptors))
            return CapabilityBundle(tool_contributions=(contrib,))

        if ref.kind == "extension-entrypoint":
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
            # not context.run_id, so an extension entrypoint nested under an
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
            ts = build_extension_entrypoint_toolset(
                self.entrypoint_resolver,
                allowed=allowed,
                allowed_kinds=allowed_kinds,
                allowed_names=tuple(allowed_names) if allowed_names else None,
                expose_call_tool=expose_call,
                max_entrypoints_per_list=cfg.get(
                    "max_entrypoints_per_extension", cap.max_entrypoints_per_extension
                ),
                emit=emit,
                executor=self.entrypoint_executor,
                parent=parent,
            )
            ekw = dict(
                source="extension",
                capability_kind="extension-entrypoint",
                capability_name=ref.name,
            )
            descs = [
                ToolDescriptor(
                    name="list_extension_entrypoints",
                    category="discovery",
                    risk="low",
                    mutating=False,
                    **ekw,
                )
            ]
            if expose_call:
                descs.append(
                    ToolDescriptor(
                        name="call_extension_entrypoint",
                        category="extension-execute",
                        risk="high",
                        mutating=True,
                        **ekw,
                    )
                )
            contrib = ToolContribution(
                tools=declared_tool_definitions(ts, tuple(descs))
            )
            return CapabilityBundle(tool_contributions=(contrib,))

        # An unknown extension-* kind slipped through; nothing to expose.
        return CapabilityBundle()
