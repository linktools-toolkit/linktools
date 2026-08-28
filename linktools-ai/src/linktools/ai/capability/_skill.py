#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill definition and function-call capability."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import SkillSpec
from ._names import SKILL_TOOL_NAMES
from ._skill_source import (
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    normalize_skill_resource_path,
)


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    spec: SkillSpec
    source_ref: "SkillSourceRef | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SkillSpec):
            raise TypeError("spec must be SkillSpec")
        if self.source_ref is not None and not isinstance(self.source_ref, SkillSourceRef):
            raise TypeError("source_ref must be SkillSourceRef or None")

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def semantic_contract(self) -> "dict[str, JsonValue]":
        contract: dict[str, JsonValue] = {
            "version": 1,
            "id": self.spec.id,
            "content": self.spec.content,
        }
        if self.spec.description is not None:
            contract["description"] = self.spec.description
        if self.source_ref is not None:
            contract["source"] = {
                "source_id": self.source_ref.source_id,
                "root": self.source_ref.root,
            }
        return contract

    @classmethod
    def from_semantic_contract(cls, contract: Mapping[str, object]) -> "SkillDefinition":
        version = contract.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        identity = contract.get("id")
        content = contract.get("content")
        description = contract.get("description")
        source = contract.get("source")
        if not isinstance(identity, str) or not identity.strip() or not isinstance(content, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if description is not None and not isinstance(description, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source_ref: SkillSourceRef | None
        if source is None:
            source_ref = None
        elif isinstance(source, Mapping):
            source_id = source.get("source_id")
            root = source.get("root")
            if not isinstance(source_id, str) or not isinstance(root, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                source_ref = SkillSourceRef(source_id, root)
            except AIError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            specification = SkillSpec(identity, content, description)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(specification, source_ref)


class SkillCapability:
    def __init__(
        self,
        skills: Sequence[SkillDefinition],
        sources: SkillSourceRegistry,
    ) -> None:
        if not isinstance(sources, SkillSourceRegistry):
            raise TypeError("sources must be SkillSourceRegistry")
        ordered = tuple(sorted(skills, key=lambda item: item.id))
        if any(not isinstance(item, SkillDefinition) for item in ordered):
            raise TypeError("skills must contain SkillDefinition values")
        ids = tuple(item.id for item in ordered)
        if len(ids) != len(set(ids)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._skills = ordered
        self._by_id = {item.id: item for item in ordered}
        self._sources = sources

    def instructions(self) -> "str | None":
        if not self._skills:
            return None
        lines = [
            "The following skills are available for this agent run.",
            "Use `load_skill` to load the instructions for a skill when it is relevant.",
        ]
        lines.extend(
            f"- {item.id}: {_skill_description(item.spec)}" for item in self._skills
        )
        return "\n".join(lines)

    async def list_skills(self) -> "list[dict[str, str]]":
        return [
            {"id": item.id, "description": _skill_description(item.spec)}
            for item in self._skills
        ]

    async def load_skill(
        self,
        skill_id: str,
        path: "str | None" = None,
    ) -> "dict[str, JsonValue]":
        definition = self._by_id.get(skill_id)
        if definition is None:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={"skill_id": skill_id},
            )
        if path is None:
            return await self._load_root(definition)
        relative = normalize_skill_resource_path(path)
        if relative == "SKILL.md":
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source_ref = definition.source_ref
        if source_ref is None:
            raise AIError(ErrorCode.ASSET_NOT_FOUND)
        source = self._sources.resolve(source_ref.source_id)
        data = await source.read(source_ref.root, relative)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AIError(
                ErrorCode.ASSET_CODEC_UNKNOWN,
                safe_details={"skill_id": skill_id, "path": relative},
            ) from error
        return {
            "id": definition.id,
            "path": relative,
            "content": content,
        }

    async def _load_root(self, definition: SkillDefinition) -> "dict[str, JsonValue]":
        result: dict[str, JsonValue] = {
            "id": definition.id,
            "description": _skill_description(definition.spec),
            "instructions": definition.spec.content,
            "resources": [],
        }
        source_ref = definition.source_ref
        if source_ref is None:
            return result
        source = self._sources.resolve(source_ref.source_id)
        view = await source.inspect(source_ref.root)
        _validate_view(view)
        result["location"] = view.location.display()
        result["resources"] = list(view.resources)
        if view.resources:
            result["usage_hint"] = _usage_hint(view)
        return result


def _skill_description(specification: SkillSpec) -> str:
    return specification.description or f"Available skill {specification.id}"


def _validate_view(view: SkillResourceView) -> None:
    if not isinstance(view, SkillResourceView):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _usage_hint(view: SkillResourceView) -> str:
    if view.location.kind == "local":
        return (
            "Resources are relative to the skill location. Resolve resource paths against `location`. "
            "When invoking a script, always use its resolved absolute path and do not rely on the current working directory."
        )
    return (
        "Resources are relative to the virtual skill location. Use `load_skill(skill_id, path)` to read them. "
        "Virtual paths are not operating-system paths and must not be passed directly to filesystem or shell tools."
    )


__all__ = ["SKILL_TOOL_NAMES", "SkillCapability", "SkillDefinition"]