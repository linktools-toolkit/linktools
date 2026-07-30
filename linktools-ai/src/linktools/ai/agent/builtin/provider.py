#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinToolProvider: resolves builtin features into sandbox-backed tools.
FunctionToolset built from the per-execution Sandbox. Returns ToolDefinitions
with proper per-tool ToolDescriptors so the governance chain
uses real categories (file-read/file-write/terminal), not conservative defaults.

Recognized refs:
  - ``builtin:file-read``    -> list_dir, read_file (read-only)
  - ``builtin:file-write``   -> write_file, batch_files, apply_patch (mutating)
  - ``builtin:terminal``     -> bash (mutating)
  - ``builtin:*``            -> all of the above (Exposure Policy still gates the
                                mutating ones behind expose_execution_tools)
  - ``builtin:file``         -> maps to file-read + file-write (subject to Exposure Policy)."""

from ...errors import AgentFeatureNotFoundError, AgentAssemblyError
from ...governance.policy.rule import RiskLevel, SideEffectKind
from ..assembly.models import AgentContribution, AgentFeatureRef
from ..assembly.provider import AgentFeatureContext
from ..tool.models import (
    ToolCategory,
    ToolDescriptor,
    ToolSource,
    declared_tool_definitions,
)
from .toolset import BuiltinToolContext, build_builtin_toolset

_WILDCARD = {"*", ""}


class BuiltinToolProvider:
    """Provides builtin file/terminal toolsets. Requires an Sandbox in
    the resolution context; a builtin ref with no backend is a configuration
    error, not a silent no-op."""

    kind = "builtin"
    supported_kinds = ("builtin",)

    async def resolve(
        self,
        ref: AgentFeatureRef,
        context: AgentFeatureContext,
    ) -> AgentContribution:
        if context.sandbox is None:
            raise AgentAssemblyError(
                f"agent {context.agent_id}: builtin:{ref.name} requires a sandbox"
            )
        enabled = _enabled_for(ref.name, agent_id=context.agent_id)
        toolset = build_builtin_toolset(
            BuiltinToolContext(sandbox=context.sandbox, enabled_tools=enabled)
        )
        descriptors = _builtin_descriptors(enabled, ref)
        return AgentContribution(
            tools=declared_tool_definitions(toolset, descriptors)
        )


def _builtin_descriptors(
    enabled: "set[str]", ref: AgentFeatureRef
) -> "tuple[ToolDescriptor, ...]":
    """Build per-tool descriptors. The Provider knows its tools' categories —
    this is a declaration, not name-based inference by the governance layer."""
    desc: "list[ToolDescriptor]" = []
    kw = dict(source=ToolSource.BUILTIN, feature=ref)
    if "file-read" in enabled:
        desc.extend(
            [
                ToolDescriptor(
                    name="list_dir",
                    category=ToolCategory.FILE_READ,
                    risk=RiskLevel.LOW,
                    side_effect=SideEffectKind.READ_ONLY,
                    **kw,
                ),
                ToolDescriptor(
                    name="read_file",
                    category=ToolCategory.FILE_READ,
                    risk=RiskLevel.LOW,
                    side_effect=SideEffectKind.READ_ONLY,
                    **kw,
                ),
            ]
        )
    if "file-write" in enabled:
        desc.extend(
            [
                ToolDescriptor(
                    name="write_file",
                    category=ToolCategory.FILE_WRITE,
                    risk=RiskLevel.MEDIUM,
                    side_effect=SideEffectKind.NAMESPACE_MUTATING,
                    **kw,
                ),
                ToolDescriptor(
                    name="batch_files",
                    category=ToolCategory.FILE_WRITE,
                    risk=RiskLevel.MEDIUM,
                    side_effect=SideEffectKind.NAMESPACE_MUTATING,
                    **kw,
                ),
                ToolDescriptor(
                    name="apply_patch",
                    category=ToolCategory.FILE_WRITE,
                    risk=RiskLevel.MEDIUM,
                    side_effect=SideEffectKind.NAMESPACE_MUTATING,
                    **kw,
                ),
            ]
        )
    if "terminal" in enabled:
        desc.append(
            ToolDescriptor(
                name="bash",
                category=ToolCategory.TERMINAL,
                risk=RiskLevel.HIGH,
                side_effect=SideEffectKind.DESTRUCTIVE,
                metadata={"requires_isolation": True, "network_access": "unknown"},
                **kw,
            )
        )
    return tuple(desc)


def _enabled_for(name: str, *, agent_id: str) -> "set[str]":
    if name in _WILDCARD:
        return {"file-read", "file-write", "terminal"}
    if name == "file-read":
        return {"file-read"}
    if name == "file-write":
        return {"file-write"}
    if name == "terminal":
        return {"terminal"}
    if name == "file":
        # Monolithic file grant: maps to read + write. A legitimate builtin ref
        # name (not a superseded alias), subject to Exposure Policy like any
        # builtin ref.
        return {"file-read", "file-write"}
    raise AgentFeatureNotFoundError(
        f"agent {agent_id}: unknown builtin feature 'builtin:{name}' "
        f"(expected 'file-read', 'file-write', 'terminal', 'file', or '*')"
    )
