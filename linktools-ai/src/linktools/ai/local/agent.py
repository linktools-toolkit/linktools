#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Local-only deterministic Agent capability assembly."""

from dataclasses import dataclass

from .project import LocalProject
from .skill import PrivateAgent, Skill


@dataclass(frozen=True, slots=True)
class LocalAgentAssembly:
    agent_id: str
    project_id: str
    skills: "tuple[Skill, ...]"
    private_agents: "tuple[PrivateAgent, ...]"
    mcp_endpoints: "tuple[str, ...]"


def assemble_agent(project: LocalProject, skills: "tuple[Skill, ...]" = ()) -> LocalAgentAssembly:
    """Assemble local capabilities in stable path order."""
    ordered = tuple(sorted(skills, key=lambda item: item.path.as_posix()))
    private = tuple(agent for skill in ordered for agent in sorted(skill.private_agents, key=lambda item: item.path.as_posix()))
    mcp = project.config.get("mcp", {})
    endpoints = tuple(sorted(mcp)) if isinstance(mcp, dict) else ()
    agent_id = str(project.config.get("default_agent", "builtin"))
    return LocalAgentAssembly(agent_id, project.project_id, ordered, private, endpoints)


__all__ = ["LocalAgentAssembly", "assemble_agent"]
