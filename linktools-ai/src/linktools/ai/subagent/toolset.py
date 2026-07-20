#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""call_subagent toolset. The tool delegates to a resolved child spec:

* ``name``/``agent_id`` -- a declared project agent (SubagentSpecProvider) or an
  extension-scoped agent (EntrypointResolver), subject to the allowlist and
  depth/timeout limits;
* ``instruction_path`` -- a skill-private agent resolved relative to the active
  skill through a ``skill_resolver`` (UnifiedSubagentResolver), with the
  permission intersection applied when the spec is built.

All three share one tool + one execution path (the executor). The
skill-private branch is only active when a ``skill_resolver`` is wired in
(otherwise ``instruction_path`` raises that skill-private subagents are not
enabled)."""

from typing import Any, Callable, Mapping

from pydantic_ai.toolsets import FunctionToolset

from ..errors import SubagentExecutionError, SubagentNotFoundError
from ..extension.entrypoint import EntrypointRef
from ..extension.scope import ExtensionScope
from ..run.identity import ParentRunIdentity
from .runner import SubagentExecutor, enforce_depth


def _parse_scope(raw: "Mapping[str, Any] | None") -> "ExtensionScope | None":
    if not raw:
        return None
    extension_id = raw.get("extension_id")
    if not extension_id:
        return None
    return ExtensionScope(
        extension_id=str(extension_id), extension_kind=raw.get("extension_kind")
    )


async def _resolve_spec(
    agent_id: str,
    scope: "ExtensionScope | None",
    subagent_provider,
    entrypoint_resolver,
):
    if scope is not None:
        if entrypoint_resolver is None:
            raise SubagentExecutionError(
                f"extension-scoped subagent {agent_id!r} needs an entrypoint resolver"
            )
        return await entrypoint_resolver.resolve_agent(
            EntrypointRef(kind="agent", name=agent_id, scope=scope)
        )
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
    max_concurrency: int = 1,
    allowed_extensions: "set[str] | None" = None,
    parent: "ParentRunIdentity | None" = None,
    skill_resolver=None,
    active_skill_provider: "Callable[[], Any] | None" = None,
    child_model_policy=None,
    parent_delegated_tools: "set[str] | None" = None,
) -> FunctionToolset:
    """Level-2 execution tool: call_subagent. Declared agent ids are admitted via
    ``name``/``agent_id``; ``instruction_path`` resolves a skill-private agent
    through ``skill_resolver``. Depth + authorization are enforced before
    delegation; concurrency is bounded by ``max_concurrency`` (per-ref)."""
    import asyncio

    toolset: FunctionToolset = FunctionToolset()
    extension_allowlist = allowed_extensions or set()
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def call_subagent(
        agent_id: "str | None" = None,
        task: "str | None" = None,
        name: "str | None" = None,
        instruction_path: "str | None" = None,
        context: "dict[str, Any] | None" = None,
        scope: "dict[str, Any] | None" = None,
    ) -> "dict[str, Any]":
        """Delegate a task to a declared subagent (name/agent_id) or a
        skill-private agent (instruction_path) and return its result."""
        if not task or not task.strip():
            raise SubagentExecutionError("call_subagent requires a non-empty 'task'")
        enforce_depth(depth_provider(), max_depth)
        ext_scope = _parse_scope(scope)
        if ext_scope is not None and ext_scope.extension_id not in extension_allowlist:
            raise SubagentNotFoundError(
                f"extension scope not allowed: {ext_scope.extension_id!r}"
            )

        if instruction_path is not None:
            spec = await _resolve_skill_private(instruction_path, task, context)
        else:
            target = name or agent_id
            if target is None:
                raise SubagentNotFoundError(
                    "call_subagent requires name or instruction_path"
                )
            if target not in allowed_names:
                raise SubagentNotFoundError(f"subagent not allowed: {target}")
            spec = await _resolve_spec(
                target, ext_scope, subagent_provider, entrypoint_resolver
            )

        if executor is None:
            raise SubagentExecutionError("no subagent executor configured")
        if parent is None:
            raise SubagentExecutionError(
                "call_subagent invoked without a parent run identity; subagent "
                "tools cannot run outside a live Run"
            )
        async with semaphore:
            result = await executor.execute(
                agent_spec=spec,
                task=task,
                context=context,
                parent=parent,
                scope=ext_scope,
                timeout_seconds=timeout_seconds,
            )
        return result.model_dump()

    async def _resolve_skill_private(instruction_path, task, context):
        if skill_resolver is None:
            raise SubagentExecutionError(
                "skill-private subagents are not enabled in this runtime"
            )
        from ..skill.private import skill_subagent_to_agent_spec
        from .skill_resolver import CallSubagentInput

        active = active_skill_provider() if active_skill_provider is not None else None
        request = CallSubagentInput(
            task=task,
            instruction_path=instruction_path,
            context=context or {},
        )
        resolved = await skill_resolver.resolve(request=request, active_skill=active)
        # Build the executable child spec with the permission intersection.
        return skill_subagent_to_agent_spec(
            resolved,
            model_policy=child_model_policy,
            parent_delegated=parent_delegated_tools,
        )

    toolset.add_function(call_subagent)
    return toolset
