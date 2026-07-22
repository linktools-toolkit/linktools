#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentProvider: the CapabilityProvider for ``subagent:<id>`` / ``subagent:*``.
Builds a call_subagent toolset scoped to the declared agent ids,
with depth/concurrency/timeout limits read from the ref config (defaults
max_depth=3, max_concurrency=1, timeout=120).

Subagents are NOT a global default tool -- the tool only exists when an agent
declares a subagent ref. Extension-scoped subagents resolve through the
EntrypointResolver; global ones through the SubagentSpecProvider."""

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from ..capability.models import CapabilityBundle
from ..capability.provider import CapabilityContext
from ..capability.models import CapabilityRef
from ..extension.resolver import EntrypointResolver
from .models import SubagentSpecProvider
from ..run.identity import ParentRunIdentity
from .runner import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DEPTH,
    DEFAULT_TIMEOUT_SECONDS,
    SubagentExecutorProtocol,
    current_depth,
)
from .toolset import build_subagent_toolset


@dataclass
class SubagentProvider:
    """CapabilityProvider for delegated subagents. ``subagent_provider`` backs
    global agents, ``entrypoint_resolver`` backs extension-scoped agents, and
    ``executor`` runs the resolved child spec (all injectable for testing)."""

    subagent_provider: "SubagentSpecProvider | None" = None
    entrypoint_resolver: "EntrypointResolver | None" = None
    executor: "SubagentExecutorProtocol | None" = None
    # Reads the contextvar so multi-hop depth accounting works when the runtime
    # executor updates it per child run.
    depth_provider: "Callable[[], int]" = field(default=current_depth)
    # Skill-private subagent support (call_subagent(instruction_path=...)).
    # All optional: when None, the tool's instruction_path branch raises that
    # skill-private subagents are not enabled (preserving the legacy behavior).
    skill_resolver: Any = None
    active_skill_provider: "Callable[[], Any] | None" = None
    child_model_policy: Any = None
    parent_delegated_tools: "set[str] | None" = None
    kind: str = "subagent"
    supported_kinds: "ClassVar[tuple[str, ...]]" = ("subagent",)

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
        # Scoped calls are confined to extensions explicitly declared on this ref.
        allowed_extensions = set(cfg.get("allowed_extensions") or [])

        # Assembly can happen outside a live run (e.g. static inspection) --
        # only build an identity when the context actually carries one.
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
        # Derive the parent's delegatable tool set from the parent agent's own
        # declared tools, so a skill-private child can never keep a tool the
        # parent lacks (the permission intersection). Only when an explicit set
        # was not configured and we can read the parent spec. On ANY failure to
        # read the parent (unknown id, e.g. a skill-private multi-hop id not in
        # the AgentCatalog, or a transient store error) FAIL CLOSED with an
        # empty set -- the child keeps no tools -- never "no constraint".
        parent_delegated = self.parent_delegated_tools
        if (
            parent_delegated is None
            and self.subagent_provider is not None
            and context.agent_id
        ):
            try:
                parent_spec = await self.subagent_provider.get(context.agent_id)
                parent_delegated = {
                    t.name for t in (getattr(parent_spec, "tools", None) or ())
                }
            except Exception:
                parent_delegated = set()

        toolset = build_subagent_toolset(
            allowed_names=allowed,
            subagent_provider=self.subagent_provider,
            entrypoint_resolver=self.entrypoint_resolver,
            executor=self.executor,
            depth_provider=self.depth_provider,
            max_depth=max_depth,
            timeout_seconds=float(timeout) if timeout is not None else None,
            max_concurrency=max_concurrency,
            allowed_extensions=allowed_extensions,
            parent=parent,
            skill_resolver=self.skill_resolver,
            active_skill_provider=self.active_skill_provider,
            child_model_policy=self.child_model_policy,
            parent_delegated_tools=parent_delegated,
        )
        from ..tool.models import ToolDescriptor
        from ..tool.models import ToolContribution, declared_tool_definitions

        descriptors = (
            ToolDescriptor(
                name="call_subagent",
                source="subagent",
                category="subagent",
                risk="medium",
                mutating=True,
                capability_kind="subagent",
                capability_name=ref.name,
            ),
        )
        contrib = ToolContribution(
            tools=declared_tool_definitions(toolset, descriptors)
        )
        return CapabilityBundle(tool_contributions=(contrib,))

    async def _allowed_names(self, ref: CapabilityRef) -> "set[str]":
        if ref.name == "*":
            if self.subagent_provider is None:
                return set()
            return set(await self.subagent_provider.list_ids())
        return {ref.name}
