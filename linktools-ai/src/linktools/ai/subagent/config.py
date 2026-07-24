#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillPrivateSubagentConfig: the typed wiring for skill-private subagents.

Skill-private subagent support (``call_subagent(instruction_path=...)`` after a
``read_skill``) needs five cooperating values that previously lived as untyped
fields on the runtime providers bundle. They are co-located here as a TYPED
struct because their types (``UnifiedSubagentResolver``, ``ActiveSkillContext``)
live in the subagent/skill domains -- referencing them from the providers
package would create a ``providers <-> {skill, subagent}`` import cycle.
Keeping the config in the subagent domain breaks that cycle: the config is
constructed by the CLI / caller and injected through ``build_runtime`` straight
into the ``SubagentProvider``, never flowing through the providers bundle.

All fields optional; an all-None config preserves legacy behavior (the
instruction_path branch raises that it is not wired when invoked).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..model.policy import ModelPolicy
from ..skill.private import ActiveSkillContext
from .skill_resolver import UnifiedSubagentResolver


@dataclass(frozen=True, slots=True)
class SkillPrivateSubagentConfig:
    """The five skill-private-subagent wiring values, typed.

    - ``skill_resolver``: a UnifiedSubagentResolver for
      ``call_subagent(instruction_path)``.
    - ``active_skill_provider``: returns the ActiveSkillContext for the current
      task (set after a successful ``read_skill``).
    - ``active_skill_lookup``: ``async (skill_id) -> ActiveSkillContext``,
      invoked by ``read_skill`` to activate a skill.
    - ``child_model_policy``: the ModelPolicy used to build the child AgentSpec
      for a skill-private subagent.
    - ``parent_delegated_tools``: optional static override of the parent's
      declared tools for the permission intersection; None means derive it
      per-resolution from the parent agent.
    """

    skill_resolver: "UnifiedSubagentResolver | None" = None
    active_skill_provider: "Callable[[], ActiveSkillContext | None] | None" = None
    active_skill_lookup: "Callable[[str], Awaitable[ActiveSkillContext]] | None" = None
    child_model_policy: "ModelPolicy | None" = None
    parent_delegated_tools: "set[str] | None" = None

    @classmethod
    def empty(cls) -> "SkillPrivateSubagentConfig":
        return cls()


__all__: "list[str]" = ["SkillPrivateSubagentConfig"]
