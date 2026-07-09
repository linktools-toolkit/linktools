#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""call_subagent toolset (spec §16.4/§16.5). The tool delegates to a resolved
child AgentSpec -- global (SubagentSpecProvider) or package-scoped
(EntrypointResolver) -- subject to an allowlist and depth/timeout limits."""

from typing import Any, Callable, Mapping

from pydantic_ai.toolsets import FunctionToolset

from ..errors import SubagentExecutionError, SubagentNotFoundError
from ..package.entrypoint import EntrypointRef
from ..package.scope import PackageScope
from .runner import SubagentExecutor, enforce_depth


def _parse_scope(raw: "Mapping[str, Any] | None") -> "PackageScope | None":
    if not raw:
        return None
    package_id = raw.get("package_id")
    if not package_id:
        return None
    return PackageScope(package_id=str(package_id), package_kind=raw.get("package_kind"))


async def _resolve_spec(
    agent_id: str,
    scope: "PackageScope | None",
    subagent_provider,
    entrypoint_resolver,
):
    if scope is not None:
        if entrypoint_resolver is None:
            raise SubagentExecutionError(
                f"package-scoped subagent {agent_id!r} needs an entrypoint resolver"
            )
        return await entrypoint_resolver.resolve_agent(
            EntrypointRef(kind="agent", name=agent_id, scope=scope))
    if subagent_provider is None:
        raise SubagentExecutionError("no subagent provider configured")
    try:
        return await subagent_provider.get(agent_id)
    except (KeyError, LookupError):
        raise SubagentNotFoundError(f"subagent not found: {agent_id}") from None


def build_subagent_toolset(
    *,
    allowed_names: "set[str]",
    subagent_provider,
    entrypoint_resolver,
    executor: "SubagentExecutor | None",
    depth_provider: "Callable[[], int]",
    max_depth: int,
    timeout_seconds: "float | None",
    allowed_packages: "set[str] | None" = None,
    parent_run_id: "str | None" = None,
    root_run_id: "str | None" = None,
    parent_session_id: "str | None" = None,
) -> FunctionToolset:
    """Level-2 execution tool: call_subagent. Only declared agent ids are
    admitted; a package-scoped call must target a package in ``allowed_packages``
    (declared-packages-only confinement); depth + authorization are enforced
    before delegation."""
    toolset: FunctionToolset = FunctionToolset()
    package_allowlist = allowed_packages or set()

    async def call_subagent(
        agent_id: str, task: str,
        context: "dict[str, Any] | None" = None,
        scope: "dict[str, Any] | None" = None,
    ) -> "dict[str, Any]":
        """Delegate a task to a declared subagent and return its result."""
        if agent_id not in allowed_names:
            raise SubagentNotFoundError(f"subagent not allowed: {agent_id}")
        enforce_depth(depth_provider(), max_depth)
        pkg_scope = _parse_scope(scope)
        # A caller-supplied package scope may only target a package this agent
        # declared; otherwise a parent could be coerced into running an
        # undeclared package's agent.
        if pkg_scope is not None and pkg_scope.package_id not in package_allowlist:
            raise SubagentNotFoundError(
                f"package scope not allowed: {pkg_scope.package_id!r}"
            )
        spec = await _resolve_spec(agent_id, pkg_scope, subagent_provider, entrypoint_resolver)
        if executor is None:
            raise SubagentExecutionError("no subagent executor configured")
        result = await executor.execute(
            agent_spec=spec, task=task, context=context,
            parent_run_id=parent_run_id, root_run_id=root_run_id,
            parent_session_id=parent_session_id, scope=pkg_scope,
            timeout_seconds=timeout_seconds,
        )
        return result.model_dump()

    toolset.add_function(call_subagent)
    return toolset
