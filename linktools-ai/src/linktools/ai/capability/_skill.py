#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Skill definition and function-call capability."""

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ToolReturn, ModelMessage, ModelRequest, ModelResponse, InstructionPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
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
    PRELOAD_SKILL_READS_METADATA_KEY,
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
        preload_skills: Sequence[SkillDefinition] = (),
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
        preload = tuple(preload_skills)
        if any(not isinstance(item, SkillDefinition) for item in preload) or len({item.id for item in preload}) != len(preload):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._preload_skills = {item.id: item for item in preload}
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
        if not self._skills and not self._preload_skills and self._discover is None:
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
        if not self._preload_skills:
            return directory
        return (directory, *(InstructionPart(content=definition.model_content, name=f"preload-skill-{hashlib.sha256(definition.id.encode()).hexdigest()}")
                             for definition in self._preload_skills.values()))

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position="innermost") if self._preload_skills else None

    def get_wrapper_model(self, model: Model) -> Model | None:
        if not self._preload_skills:
            return None
        return _PreloadSkillModel(model, self._prepare_model_request)

    def _prepare_model_request(
        self, messages: Sequence[ModelMessage], parameters: ModelRequestParameters,
    ) -> None:
        current = messages[-1] if messages else None
        if isinstance(current, ModelResponse) and current.state == "suspended":
            current = messages[-2] if len(messages) > 1 else None
        if not isinstance(current, ModelRequest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        parts = parameters.instruction_parts
        # Native continuation can restore a joined, unnamed prefix. Unset
        # parts use the request's rendering; an empty list removes it.
        instructions = current.instructions if parts is None else InstructionPart.join(parts)
        receipts = []
        for definition in self._preload_skills.values():
            expected = InstructionPart.join([InstructionPart(content=definition.model_content)]) or ""
            if expected not in (instructions or ""):
                raise AIError(ErrorCode.BINDING_CONFLICT,
                              safe_details={"reason": "preload_skill_instructions_removed", "id": definition.id})
            receipts.append(self._root_receipt(definition).to_payload())
        # Record only the request that is about to be sent, after all hooks.
        metadata = dict(current.metadata or {})
        previous = metadata.get(PRELOAD_SKILL_READS_METADATA_KEY)
        if previous is not None and previous != receipts:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata[PRELOAD_SKILL_READS_METADATA_KEY] = receipts
        current.metadata = metadata

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
        values.update(self._preload_skills)
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
        definition = self._preload_skills.get(skill_id) if path is None else None
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
        if definition.id in self._preload_skills:
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


class _PreloadSkillModel(WrapperModel):
    """Validate delivery after request hooks, including each retry's preparation."""

    def __init__(
        self, wrapped: Model,
        prepare: Callable[[Sequence[ModelMessage], ModelRequestParameters], None],
    ) -> None:
        super().__init__(wrapped)
        self._prepare = prepare

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._prepare(messages, model_request_parameters)
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: PydanticRunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self._prepare(messages, model_request_parameters)
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as response:
            yield response


__all__ = ["SkillCapability", "SkillDefinition"]
