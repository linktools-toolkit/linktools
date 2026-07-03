#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SkillSpec / SkillRegistry: skill definitions loaded from skill.md directories."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Self

from ..core.registry import BaseRegistry, BaseSpec, SpecSource, find_file
from ..support.config import (
    load_markdown_file as _load_markdown_file,
    load_markdown_text as _load_markdown_text,
)

if TYPE_CHECKING:
    from ..registry_store.store import CapabilityStore

logger = logging.getLogger("linktools.ai.skill.registry")


@dataclass(slots=True)
class SkillSpec(BaseSpec):
    description: str = ""
    source: str = ""
    category: str = ""
    metadata: "dict[str, object]" = field(default_factory=dict)
    instructions: str = ""

    @classmethod
    def from_dict(cls, payload: "Mapping[str, object]", source: SpecSource) -> Self:
        return cls(
            name=source.name,
            path=source.path,
            base_dir=source.base_dir,
            enabled=bool(payload.get("enabled", True)),
            description=str(payload.get("description") or ""),
            source=str(payload.get("source") or ""),
            category=str(payload.get("category") or ""),
            metadata=dict(payload.get("metadata", {})),
            instructions=source.instructions,
        )


class SkillRegistry(BaseRegistry[SkillSpec]):
    """Scan skill directories and load SkillSpec objects keyed by skill_id."""

    def __init__(
        self,
        *paths: Path,
        cap_store: "CapabilityStore | None" = None,
        capabilities_root: "Path | None" = None,
    ) -> None:
        super().__init__(*paths)
        self._cap_store = cap_store
        self._capabilities_root = capabilities_root
        if cap_store is not None:
            cap_store.register_primary("skill", "SKILL.md")

    async def _load(self) -> "dict[str, SkillSpec]":
        result: "dict[str, SkillSpec]" = {}
        for path in self._paths:
            if not path.is_dir():
                continue
            for child in sorted(path.iterdir()):
                spec = None
                if child.is_file() and child.name.lower() == "skill.md":
                    spec = self._load_spec(child.stem, child, None)
                elif child.is_dir():
                    markdown = find_file(child, "skill.md")
                    if markdown is not None:
                        spec = self._load_spec(child.name, markdown, child)
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("Skill '%s' from %s overrides existing registration", spec.name, child)
                    result[spec.name] = spec
                    logger.debug("skill loaded: name=%s category=%s path=%s", spec.name, spec.category, child)
        if self._cap_store is not None:
            for capability_id, _, content in await self._cap_store.iter_primaries("skill"):
                base_dir = self._capabilities_root / "skill" / capability_id if self._capabilities_root else None
                spec = self._load_spec_from_content(
                    capability_id,
                    content,
                    base_dir,
                )
                if spec and spec.enabled:
                    if spec.name in result:
                        logger.warning("Skill '%s' from DB overrides source registration", spec.name)
                    result[spec.name] = spec
                    logger.debug("skill loaded from DB: name=%s", spec.name)
        logger.debug("SkillRegistry: loaded %d skills [%s]", len(result), ", ".join(result))
        return result

    def _load_spec(self, name: str, path: Path, base_dir: "Path | None") -> SkillSpec:
        payload, instructions = _load_markdown_file(path)
        return SkillSpec.from_dict(
            payload,
            SpecSource(
                name=str(payload.get("name") or name),
                path=path,
                base_dir=base_dir,
                instructions=instructions,
            )
        )

    def _load_spec_from_content(self, capability_id: str, content: str, base_dir: "Path | None" = None) -> SkillSpec:
        payload, instructions = _load_markdown_text(content, source=f"<db:{capability_id}>")
        if base_dir is not None and not base_dir.is_dir():
            base_dir = None
        runtime_name = self._runtime_name_from_capability(capability_id, payload.get("name"), label="Skill")
        return SkillSpec.from_dict(
            payload,
            SpecSource(
                name=runtime_name,
                path=(base_dir / "SKILL.md") if base_dir else Path(f"skill/{capability_id}/SKILL.md"),
                base_dir=base_dir,
                instructions=instructions,
            ),
        )
