#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Self-contained skill directory index for a project.

A skill is a directory ``<skills_root>/<id>/`` containing ``SKILL.md`` (and
optionally ``agents/``, ``scripts/``, ``references/`` ...). This index
discovers those directories, parses each ``SKILL.md`` into a ``SkillSpec``, and
exposes the skill's root path + a content revision so an ``ActiveSkillContext``
can be minted and later validated.

The flat ``SkillCatalog`` reads ``{id}.md`` files and cannot represent a skill
directory or its ``agents/`` tree; this index is the directory-aware complement
used by :func:`linktools.ai_cli.runtime.build_cli_runtime` to back the
skill-private subagent provider."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from linktools.ai.catalog.parsing import parse_markdown_text
from linktools.ai.skill.codec import parse_skill_spec
from linktools.ai.skill.models import SkillSpec


@dataclass(frozen=True, slots=True)
class SkillInfo:
    """A discovered skill directory."""

    id: str
    root: Path
    revision: str
    spec: SkillSpec

    def list_private_agents(self) -> "list[Path]":
        """The skill's ``agents/*.md`` files (sorted), for listing/inspection.
        Absent if the skill has no ``agents/`` directory."""
        agents = self.root / "agents"
        if not agents.is_dir():
            return []
        return sorted(p for p in agents.iterdir() if p.is_file() and p.suffix == ".md")


def _skill_revision(skill_root: Path) -> str:
    """A stable revision over the skill's SKILL.md + its agents/*.md tree, so a
    change to either invalidates an active-skill context."""
    state: "list[tuple[str, int, int]]" = []
    skill_md = skill_root / "SKILL.md"
    if skill_md.is_file():
        stat = skill_md.stat()
        state.append(("SKILL.md", stat.st_mtime_ns, stat.st_size))
    agents = skill_root / "agents"
    if agents.is_dir():
        for p in sorted(agents.iterdir()):
            if p.is_file() and p.suffix == ".md":
                stat = p.stat()
                state.append((f"agents/{p.name}", stat.st_mtime_ns, stat.st_size))
    digest = sha256(repr(tuple(state)).encode("utf-8")).hexdigest()[:16]
    return digest


class DirectorySkillIndex:
    """Read-only index over ``<skills_root>/<id>/SKILL.md`` skill directories."""

    def __init__(self, skills_root: Path) -> None:
        self._root = skills_root

    @property
    def root(self) -> Path:
        return self._root

    async def list_ids(self) -> "tuple[str, ...]":
        if not self._root.is_dir():
            return ()
        ids = [
            p.name
            for p in sorted(self._root.iterdir())
            if p.is_dir() and (p / "SKILL.md").is_file()
        ]
        return tuple(ids)

    async def get(self, skill_id: str) -> "SkillInfo | None":
        skill_root = self._root / skill_id
        skill_md = skill_root / "SKILL.md"
        if not skill_md.is_file():
            return None
        text = skill_md.read_text(encoding="utf-8")
        payload, body = parse_markdown_text(text, source=str(skill_md))
        spec = parse_skill_spec(skill_id, payload, body)
        return SkillInfo(
            id=skill_id,
            root=skill_root,
            revision=_skill_revision(skill_root),
            spec=spec,
        )

    async def revision(self, skill_id: str) -> "str | None":
        info = await self.get(skill_id)
        return info.revision if info is not None else None
