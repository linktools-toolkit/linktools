#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BuiltinProvider: resolves ``builtin:file`` / ``builtin:terminal`` into the
file/terminal FunctionToolset built from the per-Run ExecutionBackend. This
moves the hardcoded ``{file, terminal}`` toolset construction out of
AgentRunner into a discoverable capability."""

from ..errors import CapabilityNotFoundError, CapabilityResolutionError
from ..execution.toolset import BuiltinToolContext, build_builtin_toolset
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
        return CapabilityBundle(toolsets=(toolset,))


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
