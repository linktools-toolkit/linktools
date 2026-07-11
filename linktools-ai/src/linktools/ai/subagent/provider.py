#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentProvider: the CapabilityProvider for ``subagent:<id>`` / ``subagent:*``.
Builds a call_subagent toolset scoped to the declared agent ids,
with depth/concurrency/timeout limits read from the ref config (defaults
max_depth=3, max_concurrency=1, timeout=120).

Subagents are NOT a global default tool -- the tool only exists when an agent
declares a subagent ref. Package-scoped subagents resolve through the
EntrypointResolver; global ones through the SubagentSpecProvider."""

from dataclasses import dataclass, field
from typing import Callable, ClassVar

from ..capability.bundle import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.ref import CapabilityRef
from ..package.resolver import EntrypointResolver
from ..providers.subagent import SubagentSpecProvider
from ..run.identity import ParentRunIdentity
from .runner import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TIMEOUT_SECONDS,
    SubagentExecutor,
    current_depth,
)
from .toolset import build_subagent_toolset


@dataclass
class SubagentProvider:
    """CapabilityProvider for delegated subagents. ``subagent_provider`` backs
    global agents, ``entrypoint_resolver`` backs package-scoped agents, and
    ``executor`` runs the resolved child spec (all injectable for testing)."""

    subagent_provider: "SubagentSpecProvider | None" = None
    entrypoint_resolver: "EntrypointResolver | None" = None
    executor: "SubagentExecutor | None" = None
    # Reads the contextvar so multi-hop depth accounting works when the runtime
    # executor updates it per child run.
    depth_provider: "Callable[[], int]" = field(default=current_depth)
    kind: str = "subagent"
    supported_kinds: "ClassVar[frozenset[str]]" = frozenset({"subagent"})

    async def resolve(
        self,
        ref: CapabilityRef,
        context: CapabilityContext,
    ) -> CapabilityBundle:
        cfg = dict(ref.config)
        max_depth = int(cfg.get("max_depth", DEFAULT_MAX_DEPTH))
        max_concurrency = int(cfg.get("max_concurrency", DEFAULT_MAX_CONCURRENCY))
        timeout = cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)

        # Reject invalid limits at resolution time.
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if timeout is not None and float(timeout) <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout}")

        allowed = await self._allowed_names(ref)
        explicit = set(cfg.get("allowed_names") or [])
        if explicit:
            allowed = allowed & explicit
        # Scoped calls are confined to packages explicitly declared on this ref.
        allowed_packages = set(cfg.get("allowed_packages") or [])

        # Assembly can happen outside a live run (e.g. static inspection) --
        # only build an identity when the context actually carries one.
        parent = None
        if context.run_id is not None and context.session_id is not None:
            parent = ParentRunIdentity(
                run_id=context.run_id, root_run_id=context.root_run_id or context.run_id,
                session_id=context.session_id, user_id=context.user_id,
                tenant_id=context.tenant_id, workspace=context.workspace,
            )
        toolset = build_subagent_toolset(
            allowed_names=allowed,
            subagent_provider=self.subagent_provider,
            entrypoint_resolver=self.entrypoint_resolver,
            executor=self.executor,
            depth_provider=self.depth_provider,
            max_depth=max_depth,
            timeout_seconds=float(timeout) if timeout is not None else None,
            max_concurrency=max_concurrency,
            allowed_packages=allowed_packages,
            parent=parent,
        )
        from ..security.descriptor import ToolDescriptor
        from ..tool.contribution import ToolContribution, declared_tool_definitions
        descriptors = (ToolDescriptor(
                name="call_subagent", source="subagent", category="subagent",
                risk="medium", mutating=True,
                capability_kind="subagent", capability_name=ref.name,
            ),)
        contrib = ToolContribution(tools=declared_tool_definitions(toolset, descriptors))
        return CapabilityBundle(tool_contributions=(contrib,))

    async def _allowed_names(self, ref: CapabilityRef) -> "set[str]":
        if ref.name == "*":
            if self.subagent_provider is None:
                return set()
            return set(await self.subagent_provider.list_ids())
        return {ref.name}
