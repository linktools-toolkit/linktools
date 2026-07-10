#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinProvider: resolves ``builtin:file`` / ``builtin:terminal`` into the
file/terminal FunctionToolset built from the per-Run ExecutionBackend. Returns
ToolContribution with proper per-tool ToolDescriptors so the governance chain
uses real categories (file-read/file-write/terminal), not conservative defaults."""

from ..errors import CapabilityNotFoundError, CapabilityResolutionError
from ..execution.toolset import BuiltinToolContext, build_builtin_toolset
from ..security.descriptor import ToolDescriptor
from ..tool.contribution import ToolContribution
from .bundle import CapabilityBundle
from .provider import CapabilityContext
from .ref import CapabilityRef

_WILDCARD = {"*", ""}


class BuiltinProvider:
    """Provides ``builtin:file`` and ``builtin:terminal`` toolsets. Requires an
    ExecutionBackend in the resolution context; a builtin ref with no backend is
    a configuration error, not a silent no-op."""

    kind = "builtin"

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        if context.execution is None:
            raise CapabilityResolutionError(
                f"agent {context.agent_id}: builtin:{ref.name} requires an execution backend"
            )
        enabled = _enabled_for(ref.name, agent_id=context.agent_id)
        toolset = build_builtin_toolset(
            BuiltinToolContext(backend=context.execution, enabled_tools=enabled)
        )
        descriptors = _builtin_descriptors(enabled, ref)
        contribution = ToolContribution(toolset=toolset, descriptors=descriptors)
        return CapabilityBundle(toolsets=(toolset,), tool_contributions=(contribution,))


def _builtin_descriptors(enabled: "set[str]", ref: CapabilityRef) -> "tuple[ToolDescriptor, ...]":
    """Build per-tool descriptors. The Provider knows its tools' categories —
    this is a declaration, not name-based inference by the governance layer."""
    desc: "list[ToolDescriptor]" = []
    kw = dict(source="builtin", capability_kind="builtin", capability_name=ref.name)
    if "file" in enabled:
        desc.extend([
            ToolDescriptor(name="list_dir", category="file-read", risk="low", mutating=False, **kw),
            ToolDescriptor(name="read_file", category="file-read", risk="low", mutating=False, **kw),
            ToolDescriptor(name="write_file", category="file-write", risk="medium", mutating=True, **kw),
            ToolDescriptor(name="batch_files", category="file-write", risk="medium", mutating=True, **kw),
            ToolDescriptor(name="apply_patch", category="file-write", risk="medium", mutating=True, **kw),
        ])
    if "terminal" in enabled:
        desc.append(ToolDescriptor(name="bash", category="terminal", risk="high", mutating=True, **kw))
    return tuple(desc)


def _enabled_for(name: str, *, agent_id: str) -> "set[str]":
    if name in _WILDCARD:
        return {"file", "terminal"}
    if name == "file":
        return {"file"}
    if name == "terminal":
        return {"terminal"}
    raise CapabilityNotFoundError(
        f"agent {agent_id}: unknown builtin capability 'builtin:{name}' "
        f"(expected 'file', 'terminal', or '*')"
    )
