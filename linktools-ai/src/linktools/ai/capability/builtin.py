#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinProvider: resolves builtin capability refs into the file/terminal
FunctionToolset built from the per-Run Sandbox. Returns
ToolContribution with proper per-tool ToolDescriptors so the governance chain
uses real categories (file-read/file-write/terminal), not conservative defaults.

Recognized refs:
  - ``builtin:file-read``    -> list_dir, read_file (read-only)
  - ``builtin:file-write``   -> write_file, batch_files, apply_patch (mutating)
  - ``builtin:terminal``     -> bash (mutating)
  - ``builtin:*``            -> all of the above (Exposure Policy still gates the
                                mutating ones behind expose_execution_tools)
  - ``builtin:file``         -> maps to file-read + file-write (subject to Exposure Policy)."""

from ..errors import CapabilityNotFoundError, CapabilityResolutionError
from ..tool.builtin.sandbox import BuiltinToolContext, build_builtin_toolset
from ..tool.models import ToolDescriptor
from ..tool.models import ToolContribution, declared_tool_definitions
from .models import CapabilityBundle
from .provider import CapabilityContext
from .models import CapabilityRef

_WILDCARD = {"*", ""}


class BuiltinProvider:
    """Provides builtin file/terminal toolsets. Requires an Sandbox in
    the resolution context; a builtin ref with no backend is a configuration
    error, not a silent no-op."""

    kind = "builtin"
    supported_kinds = ("builtin",)

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        if context.sandbox is None:
            raise CapabilityResolutionError(
                f"agent {context.agent_id}: builtin:{ref.name} requires a sandbox"
            )
        enabled = _enabled_for(ref.name, agent_id=context.agent_id)
        toolset = build_builtin_toolset(
            BuiltinToolContext(backend=context.sandbox, enabled_tools=enabled)
        )
        descriptors = _builtin_descriptors(enabled, ref)
        contribution = ToolContribution(
            tools=declared_tool_definitions(toolset, descriptors)
        )
        return CapabilityBundle(tool_contributions=(contribution,))


def _builtin_descriptors(
    enabled: "set[str]", ref: CapabilityRef
) -> "tuple[ToolDescriptor, ...]":
    """Build per-tool descriptors. The Provider knows its tools' categories —
    this is a declaration, not name-based inference by the governance layer."""
    desc: "list[ToolDescriptor]" = []
    kw = dict(source="builtin", capability_kind="builtin", capability_name=ref.name)
    if "file-read" in enabled:
        desc.extend(
            [
                ToolDescriptor(
                    name="list_dir",
                    category="file-read",
                    risk="low",
                    mutating=False,
                    **kw,
                ),
                ToolDescriptor(
                    name="read_file",
                    category="file-read",
                    risk="low",
                    mutating=False,
                    **kw,
                ),
            ]
        )
    if "file-write" in enabled:
        desc.extend(
            [
                ToolDescriptor(
                    name="write_file",
                    category="file-write",
                    risk="medium",
                    mutating=True,
                    **kw,
                ),
                ToolDescriptor(
                    name="batch_files",
                    category="file-write",
                    risk="medium",
                    mutating=True,
                    **kw,
                ),
                ToolDescriptor(
                    name="apply_patch",
                    category="file-write",
                    risk="medium",
                    mutating=True,
                    **kw,
                ),
            ]
        )
    if "terminal" in enabled:
        desc.append(
            ToolDescriptor(
                name="bash",
                category="terminal",
                risk="high",
                mutating=True,
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
    raise CapabilityNotFoundError(
        f"agent {agent_id}: unknown builtin capability 'builtin:{name}' "
        f"(expected 'file-read', 'file-write', 'terminal', 'file', or '*')"
    )
