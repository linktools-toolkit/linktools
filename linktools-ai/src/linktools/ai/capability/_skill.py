#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill definition and function-call capability."""

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler
from pydantic_ai.messages import ToolReturn, ModelRequest, ModelResponse, InstructionPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import CapabilityRef, SkillSpec, SkillSpecCodec
from ._context import AgentContext
from ._resource_path import SKILL_DECLARATION_FILES
from ._skill_source import (
    SkillLocation,
    SkillReadReceipt,
    SKILL_READ_METADATA_KEY,
    REQUIRED_SKILL_READS_METADATA_KEY,
    SkillSourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    require_skill_resource_path,
)
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._tool_metadata import tool_metadata


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
        return SkillSpecCodec().model_content(self.spec)

    @property
    def contract(self) -> "dict[str, JsonValue]":
        contract = SkillSpecCodec().to_wire_payload(self.spec)
        if self.source_ref is not None:
            contract["source"] = self.source_ref.to_payload()
        return contract

    @classmethod
    def from_contract(cls, contract: Mapping[str, object]) -> "SkillDefinition":
        source = contract.get("source")
        try:
            specification = SkillSpecCodec().from_payload(
                {key: value for key, value in contract.items() if key != "source"}
            )
        except AIError as error:
            if error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED:
                raise
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        source_ref: SkillSourceRef | None
        if source is None:
            source_ref = None
        elif isinstance(source, Mapping):
            source_ref = SkillSourceRef.from_payload(source)
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cls(specification, source_ref)


class SkillCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        skills: Sequence[SkillDefinition],
        sources: SkillSourceRegistry,
        *,
        resource_paths: Mapping[str, "str | None"] | None = None,
        discover: Callable[[], Awaitable[Sequence[SkillDefinition]]] | None = None,
        mounted_resources: Mapping[str, SkillSourceRef] | None = None,
        required_skills: Sequence[SkillDefinition] = (),
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
        required = tuple(required_skills)
        if any(not isinstance(item, SkillDefinition) for item in required) or len({item.id for item in required}) != len(required):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._required_skills = {item.id: item for item in required}
        self._skills = ordered
        self._sources = sources
        self._resource_paths = None if resource_paths is None else dict(resource_paths)
        self._mounted_resources = None if mounted_resources is None else dict(mounted_resources)
        self._discover = discover

    def get_toolset(self) -> FunctionToolset[AgentContext[object]]:
        toolset = FunctionToolset[AgentContext[object]](id=self.id)

        @toolset.tool(
            metadata=tool_metadata(
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
            metadata=tool_metadata(
                plan_safe=True,
                compaction_keep_result=True,
            )
        )
        async def load_skill(
            _ctx: PydanticRunContext[AgentContext[object]],
            skill_id: str,
            path: str | None = None,
        ) -> ToolReturn[dict[str, str | list[str]]]:
            """Load skill instructions or one relative text resource."""
            try:
                result, receipt = await self._read_skill(skill_id, path)
                return ToolReturn(return_value=result, metadata={SKILL_READ_METADATA_KEY: receipt.to_payload()})
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

    def get_instructions(self) -> str | Sequence[str | InstructionPart] | None:
        if not self._skills and not self._required_skills and self._discover is None:
            return None
        lines = [
            "The following skills are available for this agent run.",
            "Use `load_skill` to load the instructions for a skill when it is relevant.",
        ]
        if self._discover is not None:
            lines.append("Call `list_skills` to discover current skills, including skills added or edited during this run.")
        lines.extend(
            f"- {item.id}: {_skill_description(item.spec)}" for item in self._skills
        )
        directory = "\n".join(lines)
        if not self._required_skills:
            return directory
        return (directory, *(InstructionPart(content=definition.model_content, name=f"required-skill-{hashlib.sha256(definition.id.encode()).hexdigest()}")
                             for definition in self._required_skills.values()))

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position="innermost") if self._required_skills else None

    async def wrap_model_request(
        self, ctx: PydanticRunContext[AgentContext[object]], *,
        request_context: ModelRequestContext, handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        if self._required_skills:
            parts = request_context.model_request_parameters.instruction_parts
            receipts = []
            for definition in self._required_skills.values():
                name = f"required-skill-{hashlib.sha256(definition.id.encode()).hexdigest()}"
                if not any(part.name == name and part.content == definition.model_content for part in parts):
                    raise AIError(ErrorCode.BINDING_CONFLICT,
                                  safe_details={"reason": "required_skill_instructions_removed", "id": definition.id})
                receipts.append(self._root_receipt(definition).to_payload())
            # This is the actual current request, not a synthetic tool call.
            # Provider capture and the native transcript retain this metadata.
            current = request_context.messages[-1] if request_context.messages else None
            if not isinstance(current, ModelRequest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            metadata = dict(current.metadata or {})
            previous = metadata.get(REQUIRED_SKILL_READS_METADATA_KEY)
            if previous is not None and previous != receipts:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            metadata[REQUIRED_SKILL_READS_METADATA_KEY] = receipts
            current.metadata = metadata
        return await handler(request_context)

    @classmethod
    def _root_receipt(cls, definition: SkillDefinition) -> SkillReadReceipt:
        source = definition.source_ref
        declaration = None if source is None else source.declaration
        return SkillReadReceipt(
            CapabilityRef("skill", definition.id, definition.spec.revision),
            hashlib.sha256(definition.model_content.encode("utf-8")).hexdigest(),
            None if source is None else source.source_id,
            None if declaration is None else declaration.path,
            None if declaration is None else declaration.asset,
        )

    async def _current_skills(self) -> tuple[SkillDefinition, ...]:
        if self._discover is None:
            return self._skills
        values = tuple(sorted(await self._discover(), key=lambda item: item.id))
        if len({item.id for item in values}) != len(values):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        return values

    async def list_skills(self) -> "list[dict[str, str]]":
        values = {item.id: item for item in await self._current_skills()}
        values.update(self._required_skills)
        return [{"id": item.id, "description": _skill_description(item.spec)}
                for item in sorted(values.values(), key=lambda item: item.id)]

    async def load_skill(
        self,
        skill_id: str,
        path: "str | None" = None,
    ) -> "dict[str, str | list[str]]":
        result, _receipt = await self._read_skill(skill_id, path)
        return result

    async def _read_skill(
        self, skill_id: str, path: str | None,
    ) -> tuple[dict[str, str | list[str]], SkillReadReceipt]:
        definition = self._required_skills.get(skill_id) if path is None else None
        if definition is None:
            definition = next((item for item in await self._current_skills() if item.id == skill_id), None)
        if definition is None:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={"skill_id": skill_id},
            )
        if path is None:
            result = await self._load_root(definition)
            return result, self._root_receipt(definition)
        relative = require_skill_resource_path(path)
        if relative in SKILL_DECLARATION_FILES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        source_ref = definition.source_ref
        if source_ref is None:
            raise AIError(ErrorCode.ASSET_NOT_FOUND)
        source = self._sources.resolve(source_ref.source_id)
        data = await source.read(source_ref, relative)
        resource = next((item for item in source_ref.resources if item.path == relative), None)
        if resource is not None:
            resource.asset.verify_content(data)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AIError(
                ErrorCode.ASSET_CODEC_UNKNOWN,
                safe_details={"skill_id": skill_id, "path": relative},
            ) from error
        return {"id": definition.id, "path": relative, "content": content}, SkillReadReceipt(
            CapabilityRef("skill", definition.id, definition.spec.revision),
            hashlib.sha256(data).hexdigest(), source_ref.source_id, relative,
            None if resource is None else resource.asset,
        )

    async def _load_root(
        self,
        definition: SkillDefinition,
    ) -> "dict[str, str | list[str]]":
        result: dict[str, str | list[str]] = {
            "id": definition.id,
            "description": _skill_description(definition.spec),
            "instructions": definition.model_content,
            "resources": [],
        }
        source_ref = definition.source_ref
        if source_ref is None:
            return result
        source = self._sources.resolve(source_ref.source_id)
        if source_ref.declaration is not None:
            data = await source.read(source_ref, source_ref.declaration.path)
            source_ref.declaration.asset.verify_content(data)
            try:
                actual = SkillSpecCodec().model_content(SkillSpec(definition.id, data.decode("utf-8")))
            except UnicodeDecodeError as error:
                raise AIError(ErrorCode.ASSET_CODEC_UNKNOWN) from error
            if actual != definition.model_content:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if definition.id in self._required_skills:
            current = next((item for item in await self._current_skills() if item.id == definition.id), None)
            if current is None or current.source_ref is None:
                result["resource_status"] = "unavailable"
                return result
            source_ref = current.source_ref
            source = self._sources.resolve(source_ref.source_id)
            result["resource_status"] = "current"
        view = await source.inspect(source_ref)
        _validate_view(view)
        if self._resource_paths is not None:
            native_path = self._resource_paths.get(definition.id)
            if self._mounted_resources is not None:
                mounted = self._mounted_resources.get(definition.id)
                if mounted is None or (mounted.source_id, mounted.root) != (source_ref.source_id, source_ref.root):
                    native_path = None
                else:
                    mounted_by_path = {item.path: item for item in mounted.resources}
                    if any(mounted_by_path.get(item.path) != item for item in source_ref.resources):
                        native_path = None
            location = (
                SkillLocation(
                    "virtual",
                    f"{source_ref.source_id}/resources/{source_ref.root}",
                )
                if native_path is None
                else SkillLocation("local", native_path)
            )
            view = SkillSourceView(location, view.resources)
        result["location"] = view.location.display()
        result["resources"] = list(view.resources)
        if view.resources:
            result["usage_hint"] = _usage_hint(view)
        return result


def _skill_description(specification: SkillSpec) -> str:
    return specification.description or f"Available skill {specification.id}"


def _validate_view(view: SkillSourceView) -> None:
    if not isinstance(view, SkillSourceView):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _usage_hint(view: SkillSourceView) -> str:
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
