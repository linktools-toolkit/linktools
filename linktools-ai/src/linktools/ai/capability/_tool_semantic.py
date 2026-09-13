#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Tool metadata keys and strict semantic parsing."""

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from ..errors import AIError, ErrorCode

TOOL_EFFECT_METADATA_KEY = "linktools.ai.effect"
TOOL_PLAN_SAFE_METADATA_KEY = "linktools.ai.plan_safe"
TOOL_CLASS_METADATA_KEY = "linktools.ai.tool_class"
TOOL_PATH_FIELDS_METADATA_KEY = "linktools.ai.path_fields"
TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY = (
    "linktools.ai.compaction_keep_result"
)
TOOL_CONTEXT_DEDUPE_METADATA_KEY = "linktools.ai.context_dedupe"

ToolEffect = Literal["none", "replay_safe", "non_replay_safe"]
ToolClass = Literal[
    "business",
    "filesystem.read",
    "filesystem.write",
    "shell",
    "mcp",
]
ToolContextDedupe = Literal["workspace_file_read_v1"]

_TOOL_EFFECTS = {"none", "replay_safe", "non_replay_safe"}
_TOOL_CLASSES = {
    "business",
    "filesystem.read",
    "filesystem.write",
    "shell",
    "mcp",
}
_TOOL_CONTEXT_DEDUPE = "workspace_file_read_v1"


def tool_semantic_metadata(
    *,
    effect: ToolEffect | None = None,
    plan_safe: bool | None = None,
    tool_class: ToolClass | None = None,
    path_fields: Sequence[str] | None = None,
    compaction_keep_result: bool | None = None,
    context_dedupe: ToolContextDedupe | None = None,
    base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build Tool metadata while preserving unrelated upstream metadata."""
    metadata = {} if base is None else dict(base)
    if effect is not None:
        metadata[TOOL_EFFECT_METADATA_KEY] = effect
    if plan_safe is not None:
        metadata[TOOL_PLAN_SAFE_METADATA_KEY] = plan_safe
    if tool_class is not None:
        metadata[TOOL_CLASS_METADATA_KEY] = tool_class
    if path_fields is not None:
        if isinstance(path_fields, (str, bytes, bytearray)):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        metadata[TOOL_PATH_FIELDS_METADATA_KEY] = list(path_fields)
    if compaction_keep_result is not None:
        metadata[TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY] = (
            compaction_keep_result
        )
    if context_dedupe is not None:
        metadata[TOOL_CONTEXT_DEDUPE_METADATA_KEY] = context_dedupe
    validate_tool_semantic_metadata(metadata)
    return metadata


def validate_tool_semantic_metadata(
    metadata: Mapping[str, object] | None,
    *,
    require_effect: bool = False,
    require_tool_class: bool = False,
) -> None:
    """Validate LinkTools Tool metadata without coercing any value."""
    if metadata is None:
        if require_effect or require_tool_class:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        return
    if not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    effect = metadata.get(TOOL_EFFECT_METADATA_KEY)
    if effect is not None or TOOL_EFFECT_METADATA_KEY in metadata:
        if not isinstance(effect, str) or effect not in _TOOL_EFFECTS:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    elif require_effect:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    for key in (
        TOOL_PLAN_SAFE_METADATA_KEY,
        TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY,
    ):
        if key in metadata and type(metadata[key]) is not bool:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    tool_class = metadata.get(TOOL_CLASS_METADATA_KEY)
    if tool_class is not None or TOOL_CLASS_METADATA_KEY in metadata:
        if not isinstance(tool_class, str) or tool_class not in _TOOL_CLASSES:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    elif require_tool_class:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    path_fields = metadata.get(TOOL_PATH_FIELDS_METADATA_KEY)
    if path_fields is not None or TOOL_PATH_FIELDS_METADATA_KEY in metadata:
        if not isinstance(path_fields, list):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if any(
            not isinstance(field, str) or not field
            for field in path_fields
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if len(path_fields) != len(set(path_fields)):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    context_dedupe = metadata.get(TOOL_CONTEXT_DEDUPE_METADATA_KEY)
    if context_dedupe is not None or TOOL_CONTEXT_DEDUPE_METADATA_KEY in metadata:
        if context_dedupe != _TOOL_CONTEXT_DEDUPE:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def tool_effect_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    require: bool = False,
) -> ToolEffect | None:
    validate_tool_semantic_metadata(metadata, require_effect=require)
    value = None if metadata is None else metadata.get(TOOL_EFFECT_METADATA_KEY)
    return cast("ToolEffect | None", value)


def tool_plan_safe_from_metadata(metadata: Mapping[str, object] | None) -> bool:
    validate_tool_semantic_metadata(metadata)
    value = None if metadata is None else metadata.get(TOOL_PLAN_SAFE_METADATA_KEY)
    return False if value is None else cast(bool, value)


def tool_class_from_metadata(
    metadata: Mapping[str, object] | None,
) -> ToolClass | None:
    validate_tool_semantic_metadata(metadata)
    value = None if metadata is None else metadata.get(TOOL_CLASS_METADATA_KEY)
    return cast("ToolClass | None", value)


def tool_path_fields_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[str, ...]:
    validate_tool_semantic_metadata(metadata)
    value = None if metadata is None else metadata.get(TOOL_PATH_FIELDS_METADATA_KEY)
    return () if value is None else tuple(cast(list[str], value))


def tool_compaction_keep_result_from_metadata(
    metadata: Mapping[str, object] | None,
) -> bool:
    validate_tool_semantic_metadata(metadata)
    value = (
        None
        if metadata is None
        else metadata.get(TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY)
    )
    return False if value is None else cast(bool, value)


def tool_context_dedupe_from_metadata(
    metadata: Mapping[str, object] | None,
) -> ToolContextDedupe | None:
    validate_tool_semantic_metadata(metadata)
    value = (
        None
        if metadata is None
        else metadata.get(TOOL_CONTEXT_DEDUPE_METADATA_KEY)
    )
    return cast("ToolContextDedupe | None", value)


__all__ = [
    "TOOL_CLASS_METADATA_KEY",
    "TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY",
    "TOOL_CONTEXT_DEDUPE_METADATA_KEY",
    "TOOL_EFFECT_METADATA_KEY",
    "TOOL_PATH_FIELDS_METADATA_KEY",
    "TOOL_PLAN_SAFE_METADATA_KEY",
    "ToolClass",
    "ToolContextDedupe",
    "ToolEffect",
    "tool_class_from_metadata",
    "tool_compaction_keep_result_from_metadata",
    "tool_context_dedupe_from_metadata",
    "tool_effect_from_metadata",
    "tool_path_fields_from_metadata",
    "tool_plan_safe_from_metadata",
    "tool_semantic_metadata",
    "validate_tool_semantic_metadata",
]
