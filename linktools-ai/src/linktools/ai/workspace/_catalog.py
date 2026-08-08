#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Incremental local Skill and private Agent index."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from linktools.core import environ

from ..core.errors import ErrorCode, AIError
from ._root import WorkspacePolicy


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
    skill_path = Path(path).resolve()
    content = skill_path.read_text(encoding="utf-8")
    skill_id = skill_path.parent.name
    agents: list[PrivateAgent] = []
    agent_dir = skill_path.parent / "agents"
    if agent_dir.is_dir():
        for agent_path in sorted(agent_dir.glob("*.md"), key=lambda item: item.as_posix()):
            agent_content = agent_path.read_text(encoding="utf-8")
            agents.append(PrivateAgent(agent_path.stem, agent_path, agent_content, hashlib.sha256(agent_content.encode("utf-8")).hexdigest(), skill_id))
    return Skill(skill_id, skill_path, content, hashlib.sha256(content.encode("utf-8")).hexdigest(), revision, tuple(agents))

logger = environ.get_logger("ai.workspace.catalog")


class SkillIndex:
    """Refresh only changed SKILL.md files under an explicit project root."""

    def __init__(self, root: "str | Path", policy: "WorkspacePolicy | None" = None) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or WorkspacePolicy()
        self.policy.validate()
        self._fingerprints: "dict[Path, tuple[int, int, str]]" = {}
        self._skill_ids: "dict[Path, str]" = {}
        self._skills: "dict[str, Skill]" = {}
        self.revision = 0

    def refresh(self) -> int:
        candidates = (
            path for path in self.root.rglob("SKILL.md")
            if len(path.parent.relative_to(self.root).parts) <= self.policy.max_skill_depth
        )
        by_id: dict[str, list[Path]] = {}
        for path in candidates:
            by_id.setdefault(path.parent.name, []).append(path)
        selected: list[Path] = []
        for skill_id, values in sorted(by_id.items()):
            ordered = sorted(values, key=lambda item: (len(item.parent.relative_to(self.root).parts), item.as_posix()))
            nearest_depth = len(ordered[0].parent.relative_to(self.root).parts)
            nearest = tuple(item for item in ordered if len(item.parent.relative_to(self.root).parts) == nearest_depth)
            if len(nearest) > 1:
                raise AIError(ErrorCode.LOCAL_SKILL_CONFLICT, f"duplicate skill id: {skill_id}")
            selected.append(nearest[0])
        current = set(selected)
        for path in tuple(self._fingerprints):
            if path not in current:
                self.revision += 1
                skill_id = self._skill_ids.pop(path)
                self._fingerprints.pop(path)
                existing = self._skills.get(skill_id)
                if existing is not None and existing.path == path:
                    self._skills.pop(skill_id, None)
                logger.info("skill index removed path=%s revision=%s", path, self.revision)
        for path in selected:
            current.add(path)
            fingerprint = self._fingerprint(path)
            if self._fingerprints.get(path) != fingerprint:
                self.revision += 1
                skill = parse_skill(path, revision=self.revision)
                self._skills[skill.skill_id] = skill
                self._fingerprints[path] = fingerprint
                self._skill_ids[path] = skill.skill_id
                logger.info("skill index refreshed skill=%s revision=%s", skill.skill_id, self.revision)
        return self.revision

    @staticmethod
    def _fingerprint(path: Path) -> "tuple[int, int, str]":
        files = (path, *sorted((path.parent / "agents").glob("*.md"), key=lambda item: item.as_posix()))
        parts: "list[bytes]" = []
        newest = 0
        total_size = 0
        for item in files:
            stat = item.stat()
            newest = max(newest, stat.st_mtime_ns)
            total_size += stat.st_size
            parts.extend((item.name.encode("utf-8"), b"\0", item.read_bytes(), b"\0"))
        return newest, total_size, hashlib.sha256(b"".join(parts)).hexdigest()

    def resolve(self, skill_id: str) -> Skill:
        return self._skills[skill_id]

    def resolve_agent(self, skill_id: str, agent_id: str) -> PrivateAgent:
        skill = self.resolve(skill_id)
        for agent in skill.private_agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(agent_id)

    def list(self) -> "tuple[Skill, ...]":
        return tuple(self._skills[key] for key in sorted(self._skills))


__all__ = ["PrivateAgent", "Skill", "SkillIndex", "parse_skill"]
