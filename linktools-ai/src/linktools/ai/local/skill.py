#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SKILL.md and private subagent parsing."""

from dataclasses import dataclass
from pathlib import Path

from ..foundation.digest import sha256_digest


@dataclass(frozen=True, slots=True)
class PrivateAgent:
    agent_id: str
    path: Path
    content: str
    digest: str
    skill_id: str


@dataclass(frozen=True, slots=True)
class Skill:
    skill_id: str
    path: Path
    content: str
    digest: str
    revision: int
    private_agents: "tuple[PrivateAgent, ...]" = ()


def parse_skill(path: "str | Path", *, revision: int = 1) -> Skill:
    """Parse one skill file and its sibling private agents."""
    skill_path = Path(path).resolve()
    content = skill_path.read_text(encoding="utf-8")
    skill_id = skill_path.parent.name
    agents = []
    agent_dir = skill_path.parent / "agents"
    if agent_dir.is_dir():
        for agent_path in sorted(agent_dir.glob("*.md"), key=lambda item: item.as_posix()):
            agent_content = agent_path.read_text(encoding="utf-8")
            agents.append(PrivateAgent(agent_path.stem, agent_path, agent_content, sha256_digest(agent_content.encode("utf-8")), skill_id))
    return Skill(skill_id, skill_path, content, sha256_digest(content.encode("utf-8")), revision, tuple(agents))


__all__ = ["PrivateAgent", "Skill", "parse_skill"]
