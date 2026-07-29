#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExtensionProvider: the AgentFeatureProvider for ``extension`` / ``extension-asset``
/ ``extension-entrypoint`` tool refs.

- ``extension:<id>``            -> prompt catalog only (Level 0), no tools.
- ``extension-asset:<id>``   -> Level-1 list/read asset tools for that extension.
- ``extension-entrypoint:<id>`` -> Level-1 list-entrypoints tool (+ opt-in call).

An extension NEVER auto-exposes all its assets/entrypoints as tools: only the
explicitly declared extension ids become reachable, and only the read/list tools
are added by default."""

from dataclasses import dataclass
from typing import ClassVar


from ..assembly.models import AgentContribution, AgentFeatureRef
from ..assembly.provider import AgentFeatureContext
from .spec import ExtensionContentSource
from ...execution.identity import ParentRunIdentity
from ...governance.policy.rule import RiskLevel, SideEffectKind
from ..tool.models import (
    ToolCategory,
    ToolDescriptor,
    ToolSource,
    declared_tool_definitions,
)
from ..subagent.runner import SubagentExecutorProtocol
from .resolver import EntrypointResolver
from .scope import ExtensionScope
from .toolset import build_extension_entrypoint_toolset, build_extension_resource_toolset


@dataclass
class ExtensionProvider:
    """AgentFeatureProvider for extension-scoped features. Depends only on the
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
        ref: AgentFeatureRef,
        context: AgentFeatureContext,
    ) -> AgentContribution:
        # ``extension:<id>`` (this provider's own kind) -> prompt catalog only.
        if ref.kind == "extension":
            return AgentContribution(
                prompt_sections={
                    "extensions": f"Extension declared: {ref.name}. Use extension-asset / "
                    f"extension-entrypoint tools to inspect it when enabled.",
                }
            )

        scope = ExtensionScope(extension_id=ref.name)
        allowed: "dict[str, ExtensionScope]" = {ref.name: scope}
        cfg = dict(ref.config)
        emit = None

        if ref.kind == "extension-asset":
            if self.content_source is None:
                return AgentContribution()
            ts = build_extension_resource_toolset(
                self.content_source,
                allowed=allowed,
                max_resources_per_list=cfg.get(
                    "max_resources_per_list", 50
                ),
                max_read_bytes=cfg.get("max_read_bytes", 65536),
                emit=emit,
            )
            pkw = dict(
                source=ToolSource.EXTENSION,
                feature=ref,
                category=ToolCategory.EXTENSION_READ,
                risk=RiskLevel.LOW,
                side_effect=SideEffectKind.READ_ONLY,
            )
            descriptors = (
                ToolDescriptor(name="list_extension_content", **pkw),
                ToolDescriptor(name="read_extension_content", **pkw),
            )
            return AgentContribution(
                tools=declared_tool_definitions(ts, descriptors)
            )

        if ref.kind == "extension-entrypoint":
            if self.entrypoint_resolver is None:
                return AgentContribution()
            allowed_kinds = tuple(
                cfg.get("allowed_kinds", ("agent",))
            )
            allowed_names = cfg.get("allowed_names")
            expose_call = (
                bool(cfg.get("expose_call_tool", False))
            )
            # Same ParentRunIdentity shape every spawner builds -- root_execution_id
            # comes from context.root_execution_id (the ACTUAL root of the chain),
            # not context.run_id, so an extension entrypoint nested under an
            # existing subagent chain doesn't truncate lineage to itself.
            parent = None
            if context.execution_id and context.session_id:
                parent = ParentRunIdentity(
                    run_id=context.execution_id,
                    root_execution_id=context.root_execution_id,
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
                    "max_entrypoints_per_extension", 20
                ),
                emit=emit,
                executor=self.entrypoint_executor,
                parent=parent,
            )
            ekw = dict(
                source=ToolSource.EXTENSION,
                feature=ref,
            )
            descs = [
                ToolDescriptor(
                    name="list_extension_entrypoints",
                    category=ToolCategory.DISCOVERY,
                    risk=RiskLevel.LOW,
                    side_effect=SideEffectKind.READ_ONLY,
                    **ekw,
                )
            ]
            if expose_call:
                descs.append(
                    ToolDescriptor(
                        name="call_extension_entrypoint",
                        category=ToolCategory.EXTENSION_EXECUTE,
                        risk=RiskLevel.HIGH,
                        side_effect=SideEffectKind.NAMESPACE_MUTATING,
                        **ekw,
                    )
                )
            return AgentContribution(
                tools=declared_tool_definitions(ts, tuple(descs))
            )

        # An unknown extension-* kind slipped through; nothing to expose.
        return AgentContribution()
