#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Incremental local Skill and private Agent index."""

import hashlib
from pathlib import Path

from linktools.core import environ

from .skill import PrivateAgent, Skill, parse_skill
from .config import LocalPolicy

logger = environ.get_logger("ai.local.index")


class SkillIndex:
    """Refresh only changed SKILL.md files under an explicit project root."""

    def __init__(self, root: "str | Path", policy: "LocalPolicy | None" = None) -> None:
        self.root = Path(root).resolve()
        self.policy = policy or LocalPolicy()
        self.policy.validate()
        self._fingerprints: "dict[Path, tuple[int, int, str]]" = {}
        self._skill_ids: "dict[Path, str]" = {}
        self._skills: "dict[str, Skill]" = {}
        self.revision = 0

    def refresh(self) -> int:
        current: "set[Path]" = set()
        paths = (
            path for path in self.root.rglob("SKILL.md")
            if len(path.parent.relative_to(self.root).parts) <= self.policy.max_skill_depth
        )
        for path in sorted(paths, key=lambda item: item.as_posix()):
            current.add(path)
            fingerprint = self._fingerprint(path)
            if self._fingerprints.get(path) != fingerprint:
                self.revision += 1
                skill = parse_skill(path, revision=self.revision)
                self._skills[skill.skill_id] = skill
                self._fingerprints[path] = fingerprint
                self._skill_ids[path] = skill.skill_id
                logger.info("skill index refreshed skill=%s revision=%s", skill.skill_id, self.revision)
        for path in tuple(self._fingerprints):
            if path not in current:
                self.revision += 1
                self._fingerprints.pop(path)
                self._skills.pop(self._skill_ids.pop(path), None)
                logger.info("skill index removed path=%s revision=%s", path, self.revision)
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


__all__ = ["SkillIndex"]
