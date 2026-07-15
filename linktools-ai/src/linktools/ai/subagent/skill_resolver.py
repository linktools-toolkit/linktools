#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified subagent resolution for the CLI.

A single ``call_subagent`` tool serves two resolution modes:

* ``name`` -- a project-level agent (resolved through the project AgentRegistry);
* ``instruction_path`` -- a skill-private agent (resolved relative to the active
  skill through :class:`SkillSubagentProvider`).

This module holds the request model, the skill-private provider, and the
unified resolver that picks the branch. It composes the security primitives in
:mod:`linktools.ai.skill.private` (path/symlink rejection, parsing) and an
injectable project-agent source, so it is fully unit-testable without a live
Runtime. The live ``call_subagent`` tool wires this resolver in at assembly
time; resolution itself is pure."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from ..errors import SubagentResolutionError
from ..skill.private import (
    ActiveSkillContext,
    SkillSubagentSpec,
    parse_skill_subagent,
    resolve_skill_agent_path,
)


@dataclass(frozen=True, slots=True)
class CallSubagentInput:
    """The ``call_subagent`` request model.

    Exactly one of ``name`` / ``instruction_path`` is required (validated)."""

    task: str
    name: "str | None" = None
    instruction_path: "str | None" = None
    context: "Mapping[str, Any]" = field(default_factory=dict)
    include_parent_summary: bool = False

    def validate(self) -> None:
        if (self.name is None) == (self.instruction_path is None):
            raise SubagentResolutionError(
                "exactly one of name or instruction_path is required"
            )
        if not self.task.strip():
            raise SubagentResolutionError("task must not be blank")


class _ProjectAgentSource(Protocol):
    """Anything that returns an AgentSpec by name (the project AgentRegistry
    satisfies this). Used for the ``name`` resolution branch."""

    async def get(self, agent_id: str) -> Any: ...


class _SkillIndex(Protocol):
    """Read-only view over a project's self-contained skill directories.

    ``get`` returns a ``SkillInfo`` (carrying the skill root + revision) or
    None; ``revision`` returns the skill's current revision or None."""

    async def get(self, skill_id: str) -> Any: ...


class SkillSubagentProvider:
    """Resolves a skill-private agent from an active skill.

    Reuses ``resolve_skill_agent_path`` (security) + ``parse_skill_subagent``
    (parsing). Fails if the active skill no longer exists or changed revision
    on disk since it was activated -- so a stale active-skill context cannot
    address a skill that was edited/removed underneath it."""

    def __init__(self, *, skills: "_SkillIndex", default_timeout_seconds: int) -> None:
        self._skills = skills
        self._default_timeout_seconds = default_timeout_seconds

    async def resolve(
        self,
        *,
        active_skill: ActiveSkillContext,
        instruction_path: str,
    ) -> SkillSubagentSpec:
        skill = await self._skills.get(active_skill.skill_id)
        if skill is None:
            raise SubagentResolutionError("active skill does not exist")
        if skill.revision != active_skill.revision:
            raise SubagentResolutionError("active skill revision changed")
        path = resolve_skill_agent_path(
            skill_root=active_skill.skill_root,
            instruction_path=instruction_path,
        )
        return parse_skill_subagent(
            skill_id=active_skill.skill_id,
            instruction_path=instruction_path,
            path=path,
            default_timeout_seconds=self._default_timeout_seconds,
        )


class UnifiedSubagentResolver:
    """Picks the project vs skill-private branch for a ``call_subagent`` request
    . Validates the request, then dispatches."""

    def __init__(
        self,
        *,
        project_agents: "_ProjectAgentSource",
        skill_agents: SkillSubagentProvider,
    ) -> None:
        self._project_agents = project_agents
        self._skill_agents = skill_agents

    async def resolve(
        self,
        *,
        request: CallSubagentInput,
        active_skill: "ActiveSkillContext | None",
    ):
        request.validate()
        if request.name is not None:
            from ..errors import RegistryNotFoundError

            try:
                spec = await self._project_agents.get(request.name)
            except RegistryNotFoundError as exc:
                raise SubagentResolutionError(
                    f"unknown subagent: {request.name}"
                ) from exc
            if spec is None:
                raise SubagentResolutionError(f"unknown subagent: {request.name}")
            return spec
        if active_skill is None:
            raise SubagentResolutionError("instruction_path requires an active skill")
        return await self._skill_agents.resolve(
            active_skill=active_skill,
            instruction_path=request.instruction_path,
        )
