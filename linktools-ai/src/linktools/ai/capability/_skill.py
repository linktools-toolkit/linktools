#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill definition and function-call capability."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core import JsonValue
from ..asset import AssetVersionRef
from ..errors import AIError, ErrorCode
from ..spec import SkillMarkdownSpecCodec, SkillSpec
from ._context import AgentContext
from ._skill_source import (
    SkillLocation,
    SkillResourceVersion,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    normalize_skill_resource_path,
)
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._tool_semantic import tool_semantic_metadata


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
    def model_content(self) -> str:
        return SkillMarkdownSpecCodec().model_content(self.spec.content)

    @property
    def contract(self) -> "dict[str, JsonValue]":
        contract: dict[str, JsonValue] = {
            "version": 1,
            "id": self.spec.id,
            "revision": self.spec.revision,
            "content": self.spec.content,
        }
        if self.spec.description is not None:
            contract["description"] = self.spec.description
        if self.spec.metadata:
            contract["metadata"] = dict(self.spec.metadata)
        if self.source_ref is not None:
            source: dict[str, JsonValue] = {
                "source_id": self.source_ref.source_id,
                "root": self.source_ref.root,
            }
            if self.source_ref.resource_versions:
                source["resource_versions"] = [
                    {
                        "path": item.path,
                        "asset": item.asset.to_payload(),
                        "executable_bits": item.executable_bits,
                    }
                    for item in self.source_ref.resource_versions
                ]
            contract["source"] = source
        return contract

    @classmethod
    def from_contract(cls, contract: Mapping[str, object]) -> "SkillDefinition":
        version = contract.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        identity = contract.get("id")
        revision = contract.get("revision", 1)
        content = contract.get("content")
        description = contract.get("description")
        metadata = contract.get("metadata", {})
        source = contract.get("source")
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
            or not isinstance(content, str)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if description is not None and not isinstance(description, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source_ref: SkillSourceRef | None
        if source is None:
            source_ref = None
        elif isinstance(source, Mapping):
            source_id = source.get("source_id")
            root = source.get("root")
            raw_versions = source.get("resource_versions")
            versions: tuple[SkillResourceVersion, ...] = ()
            if raw_versions is not None:
                if not isinstance(raw_versions, list):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                parsed: list[SkillResourceVersion] = []
                try:
                    for raw in raw_versions:
                        if not isinstance(raw, Mapping):
                            raise ValueError
                        path = raw.get("path")
                        asset = raw.get("asset")
                        mode = raw.get("executable_bits", 0)
                        if (
                            not isinstance(path, str)
                            or isinstance(mode, bool)
                            or not isinstance(mode, int)
                        ):
                            raise ValueError
                        parsed.append(
                            SkillResourceVersion(
                                path,
                                AssetVersionRef.from_payload(asset),
                                mode,
                            )
                        )
                except (TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                versions = tuple(sorted(parsed, key=lambda item: item.path))
            try:
                source_ref = SkillSourceRef(
                    source_id,
                    root,
                    versions,
                )
            except AIError as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            specification = SkillSpec(
                identity,
                content,
                description,
                metadata,
                revision=revision,
            )
        except (TypeError, ValueError, UnicodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return cls(specification, source_ref)


class SkillCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        skills: Sequence[SkillDefinition],
        sources: SkillSourceRegistry,
        *,
        resource_paths: Mapping[str, "str | None"] | None = None,
        preloaded_skill_ids: Sequence[str] = (),
        max_preloaded_bytes: int = 256 * 1024,
    ) -> None:
        self.id = "linktools.ai.skills"
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
        self._resource_paths = dict(resource_paths or {})
        preload_ids = tuple(sorted(preloaded_skill_ids))
        if len(preload_ids) != len(set(preload_ids)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        if any(skill_id not in self._by_id for skill_id in preload_ids):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if (
            not isinstance(max_preloaded_bytes, int)
            or isinstance(max_preloaded_bytes, bool)
            or max_preloaded_bytes < 0
        ):
            raise ValueError("max_preloaded_bytes must be a non-negative integer")
        self._preloaded = tuple(self._by_id[skill_id] for skill_id in preload_ids)
        self._max_preloaded_bytes = max_preloaded_bytes

    def get_instructions(self) -> str | None:
        return self.instructions()

    def get_toolset(self) -> FunctionToolset[AgentContext[object]]:
        toolset = FunctionToolset[AgentContext[object]](id=self.id)

        @toolset.tool(
            metadata=tool_semantic_metadata(
                plan_safe=True,
                compaction_keep_result=True,
            )
        )
        async def list_skills(
            _ctx: PydanticRunContext[AgentContext[object]],
        ) -> list[dict[str, str]]:
            """List skills available for this agent run."""
            return await self.list_skills()

        @toolset.tool(
            metadata=tool_semantic_metadata(
                plan_safe=True,
                compaction_keep_result=True,
            )
        )
        async def load_skill(
            _ctx: PydanticRunContext[AgentContext[object]],
            skill_id: str,
            path: str | None = None,
        ) -> dict[str, str | list[str]]:
            """Load skill instructions or one relative text resource."""
            try:
                return cast(
                    dict[str, str | list[str]],
                    await self.load_skill(skill_id, path),
                )
            except AIError as error:
                if error.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID:
                    raise ToolCallRetry(
                        "The requested skill id is not available. Call list_skills and "
                        "retry with one of the returned skill ids."
                    ) from error
                if error.code is ErrorCode.REQUEST_FIELD_INVALID:
                    raise ToolCallRetry(
                        "The skill resource path is invalid. Load the skill root and "
                        "retry with one of its listed resource paths."
                    ) from error
                if error.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT:
                    raise ToolCallFailed(
                        "The requested skill resource is outside the skill root and "
                        "cannot be accessed. Use a listed resource inside the skill "
                        "root or continue without it."
                    ) from error
                if error.code is ErrorCode.ASSET_NOT_FOUND:
                    if path is None:
                        message = (
                            "The requested skill resource root is unavailable. Repeating "
                            "the same load_skill call will not resolve it; use another "
                            "skill or continue without it."
                        )
                    else:
                        message = (
                            "The requested skill resource does not exist. Load the skill "
                            "root and choose one of its listed resources, or continue "
                            "without it."
                        )
                    raise ToolCallFailed(message) from error
                if error.code is ErrorCode.ASSET_CODEC_UNKNOWN:
                    raise ToolCallFailed(
                        "The requested skill resource is not UTF-8 text and cannot be "
                        "loaded by load_skill. Use another resource or another tool."
                    ) from error
                raise

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

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
        if self._preloaded:
            lines.extend(
                (
                    "The following skill instructions are preloaded for this agent run.",
                    *(
                        f"[skill: {item.id}]\n{item.model_content}"
                        for item in self._preloaded
                    ),
                )
            )
            try:
                preloaded = "\n\n".join(
                    f"[skill: {item.id}]\n{item.model_content}"
                    for item in self._preloaded
                ).encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
            if len(preloaded) > self._max_preloaded_bytes:
                raise AIError(ErrorCode.PROMPT_TOO_LARGE)
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
            "instructions": definition.model_content,
            "resources": [],
        }
        source_ref = definition.source_ref
        if source_ref is None:
            return result
        source = self._sources.resolve(source_ref.source_id)
        view = await source.inspect(source_ref.root)
        _validate_view(view)
        if definition.id in self._resource_paths:
            native_path = self._resource_paths[definition.id]
            location = (
                SkillLocation(
                    "virtual",
                    f"{source_ref.source_id}/skills/{source_ref.root}",
                )
                if native_path is None
                else SkillLocation("local", native_path)
            )
            view = SkillResourceView(location, view.resources)
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


__all__ = ["SkillCapability", "SkillDefinition"]
