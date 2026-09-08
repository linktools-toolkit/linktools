#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-scoped attachment projection at the public Pydantic model hook."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.messages import ModelMessage, ModelRequest, TextContent, UserContent, UserPromptPart

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ._attachment_adapter import (
    AttachmentRequestModel,
    _MARKER_KEY,
    _marker_matches,
    _semantic_entry_digest,
    attachment_placeholder,
)
from ._input import _decode_user_content
from .state import (
    AttachmentEntry,
    ContentRef,
    InputAttachmentPart,
    InputNativePart,
    InputTextPart,
    InputV2,
    Locator,
    ModelExposure,
    ModelExposureEntry,
    PathOrigin,
    semantic_attachment_entry,
)

_ExecutionCheck = Callable[[int], Awaitable[None]]
_EntryAuthorization = Callable[[int, tuple[ModelExposureEntry, ...]], Awaitable[None]]
_ExposureCommit = Callable[[int, tuple[ModelExposureEntry, ...]], Awaitable[ModelExposure]]
_ContentRead = Callable[[ContentRef], Awaitable[bytes]]


class AttachmentProjectionCapability(AbstractCapability[Any]):
    """Install one attachment WrapperModel for each logical model request step."""

    def __init__(
        self,
        *,
        execution_id: str,
        step_run_id: str,
        path_origin: PathOrigin,
        activations: Sequence[ModelExposureEntry],
        check_execution: _ExecutionCheck,
        authorize_entries: _EntryAuthorization,
        commit_exposure: _ExposureCommit,
        read_content: _ContentRead,
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution_id is required")
        if not isinstance(step_run_id, str) or not step_run_id:
            raise ValueError("step_run_id is required")
        if not isinstance(path_origin, PathOrigin):
            raise TypeError("path_origin must be PathOrigin")
        values = tuple(activations)
        if any(not isinstance(value, ModelExposureEntry) for value in values):
            raise TypeError("activations must contain ModelExposureEntry values")
        if len({value.activation_id for value in values}) != len(values):
            raise ValueError("activations contain duplicate activation ids")
        self._execution_id = execution_id
        self._step_run_id = step_run_id
        self._path_origin = path_origin
        self._activations = values
        self._check_execution = check_execution
        self._authorize_entries = authorize_entries
        self._commit_exposure = commit_exposure
        self._read_content = read_content

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = _prepare_current_messages(
            request_context.messages,
            self._activations,
        )
        model = request_context.model
        if isinstance(model, AttachmentRequestModel):
            model = model.wrapped
        wrapped = AttachmentRequestModel(
            model,
            execution_id=self._execution_id,
            step_run_id=self._step_run_id,
            run_step=ctx.run_step,
            path_origin=self._path_origin,
            activations=self._activations,
            check_execution=self._check_execution,
            authorize_entries=self._authorize_entries,
            commit_exposure=self._commit_exposure,
            read_content=self._read_content,
        )
        return replace(
            request_context,
            messages=messages,
            model=wrapped,
        )


def initial_attachment_prompt(
    prompt: InputV2,
    manifest: Sequence[AttachmentEntry],
    *,
    execution_id: str,
    execution_record_key: str,
) -> tuple[str | tuple[UserContent, ...], tuple[ModelExposureEntry, ...]]:
    """Project admitted InputV2 into lightweight direct slots and active entries."""
    if not isinstance(prompt, InputV2):
        raise TypeError("prompt must be InputV2")
    entries = tuple(manifest)
    if any(not isinstance(value, AttachmentEntry) for value in entries):
        raise TypeError("manifest must contain AttachmentEntry values")
    if not isinstance(execution_id, str) or not execution_id:
        raise ValueError("execution_id is required")
    source = Locator("state:execution", "records", execution_record_key)
    projected: list[UserContent] = []
    active: list[ModelExposureEntry] = []
    active_by_slot: dict[int, ModelExposureEntry] = {}
    for part in prompt.parts:
        if isinstance(part, InputTextPart):
            projected.append(part.text)
            continue
        if isinstance(part, InputNativePart):
            projected.extend(_decode_user_content(dict(part.value)))
            continue
        if not isinstance(part, InputAttachmentPart) or part.index >= len(entries):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        activation = active_by_slot.get(part.index)
        if activation is None:
            entry = entries[part.index]
            semantic_digest = _semantic_entry_digest(semantic_attachment_entry(entry))
            activation = ModelExposureEntry(
                canonical_sha256(
                    {
                        "version": 1,
                        "execution_id": execution_id,
                        "source": source.to_json(),
                        "slot": part.index,
                        "entry_digest": semantic_digest,
                    }
                ),
                source,
                part.index,
                entry,
            )
            active_by_slot[part.index] = activation
            active.append(activation)
        projected.append(attachment_placeholder(activation))
    if len(projected) == 1 and isinstance(projected[0], str):
        return projected[0], tuple(active)
    return tuple(projected), tuple(active)


def _prepare_current_messages(
    messages: Sequence[ModelMessage],
    activations: Sequence[ModelExposureEntry],
) -> list[ModelMessage]:
    current = {value.activation_id: value for value in activations}
    present: set[str] = set()
    normalized: list[ModelMessage] = []
    last_request = -1
    for index, message in enumerate(messages):
        if not isinstance(message, ModelRequest):
            normalized.append(message)
            continue
        last_request = index
        parts = []
        changed = False
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                parts.append(part)
                continue
            content: list[UserContent] = []
            part_changed = False
            for item in part.content:
                if not isinstance(item, TextContent) or not isinstance(item.metadata, Mapping):
                    content.append(item)
                    continue
                marker = item.metadata.get(_MARKER_KEY)
                if not isinstance(marker, Mapping):
                    content.append(item)
                    continue
                activation_id = marker.get("activation_id")
                activation = current.get(activation_id) if isinstance(activation_id, str) else None
                if activation is None:
                    metadata = dict(item.metadata)
                    metadata.pop(_MARKER_KEY, None)
                    content.append(replace(item, metadata=metadata or None))
                    part_changed = True
                    continue
                if not _marker_matches(marker, activation):
                    raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
                present.add(activation.activation_id)
                content.append(item)
            if part_changed:
                parts.append(replace(part, content=tuple(content)))
                changed = True
            else:
                parts.append(part)
        normalized.append(replace(message, parts=tuple(parts)) if changed else message)
    missing = [value for value in activations if value.activation_id not in present]
    if not missing:
        return normalized
    if last_request < 0 or last_request >= len(normalized):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    request = normalized[last_request]
    if not isinstance(request, ModelRequest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    normalized[last_request] = replace(
        request,
        parts=(
            *request.parts,
            UserPromptPart(
                content=tuple(attachment_placeholder(value) for value in missing)
            ),
        ),
    )
    return normalized


__all__: tuple[str, ...] = ()
