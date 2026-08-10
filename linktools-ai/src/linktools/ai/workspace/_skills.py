#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-local Agent Skill discovery."""

import unicodedata
from pathlib import Path

import yaml

from ..capability import SkillCatalogSnapshot, SkillDescriptor
from ..core import canonical_sha256
from ..spec import SkillSpec


def load_local_skill_catalog(root: Path) -> SkillCatalogSnapshot:
    descriptors: list[SkillDescriptor] = []
    specifications: list[SkillSpec] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir():
            continue
        skill_file = directory / "SKILL.md"
        if not skill_file.is_file():
            continue
        skill_id, description, content = _read_skill(skill_file)
        revision = int(canonical_sha256(["workspace-skill", skill_id, content]), 16)
        descriptors.append(SkillDescriptor(skill_id, revision, description))
        specifications.append(SkillSpec(skill_id, revision, content))
    return SkillCatalogSnapshot(tuple(descriptors), tuple(specifications))


def _read_skill(path: Path) -> "tuple[str, str, str]":
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path} must start with YAML frontmatter delimited by `---`.")
    closing = next((index for index, line in enumerate(lines[1:], start=1) if line == "---"), None)
    if closing is None:
        raise ValueError(f"{path} has unclosed YAML frontmatter.")
    metadata = yaml.safe_load("\n".join(lines[1:closing]))
    if not isinstance(metadata, dict):
        raise TypeError(f"YAML frontmatter in {path} must be a mapping.")
    skill_id = metadata.get("name", path.parent.name)
    description = metadata.get("description")
    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError(f"Skill name in {path} must be a non-empty string.")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill description in {path} must be a non-empty string.")
    normalized_id = _normalize_skill_id(skill_id, path)
    if normalized_id != _normalize_skill_id(path.parent.name, path):
        raise ValueError(f"Skill name {skill_id!r} in {path} must match its parent directory.")
    content_lines = lines[closing + 1 :]
    while content_lines and not content_lines[0].strip():
        content_lines.pop(0)
    while content_lines and not content_lines[-1].strip():
        content_lines.pop()
    return normalized_id, description.strip(), "\n".join(content_lines)


def _normalize_skill_id(value: str, path: Path) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    if (
        not normalized
        or len(normalized) > 64
        or normalized != normalized.lower()
        or normalized.startswith("-")
        or normalized.endswith("-")
        or "--" in normalized
        or not all(character.isalnum() or character == "-" for character in normalized)
    ):
        raise ValueError(f"Invalid skill name {value!r} in {path}.")
    return normalized


__all__ = ["load_local_skill_catalog"]
