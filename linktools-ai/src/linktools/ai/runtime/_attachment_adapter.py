#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic model projection for Runtime-managed attachments."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..errors import AIError, ErrorCode
from ._object import read_runtime_object
from .state import (
    ContentRef,
    ModelExposure,
    ModelExposureEntry,
    PathOrigin,
    RuntimeDomain,
    RuntimeState,
)

_MARKER_KEY = "linktools.ai.attachment.v1"
_EXPOSURE_METADATA_KEY = "linktools.ai.exposure_id"

_ExecutionCheck = Callable[[int], Awaitable[None]]
_EntryAuthorization = Callable[[int, tuple[ModelExposureEntry, ...]], Awaitable[None]]
_ExposureCommit = Callable[[int, tuple[ModelExposureEntry, ...]], Awaitable[ModelExposure]]
_ContentRead = Callable[[ContentRef], Awaitable[bytes]]


class AttachmentContentResolver:
    """Resolve a ContentRef through its Runtime domain and owner scope."""

    def __init__(self, state: RuntimeState) -> None:
        self._state = state

    async def read(self, content: ContentRef) -> bytes:
        if not isinstance(content, ContentRef):
            raise TypeError("content must be ContentRef")
        try:
            domain = RuntimeDomain(content.domain)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        store = self._state.working_object_store(
            domain,
            owner_scope=content.owner_scope,
        )
        return await read_runtime_object(store, content.object)


@dataclass(slots=True)
class _ExposureTicket:
    run_step: int
    entries: tuple[ModelExposureEntry, ...]
    exposure: ModelExposure | None = None
    bodies: dict[str, bytes] | None = None
    suspended: bool = False


class AttachmentModel(WrapperModel):
    """Expand trusted attachment placeholders only at the model boundary."""

    def __init__(
        self,
        wrapped: Model,
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
        super().__init__(wrapped)
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
        self._activations = {value.activation_id: value for value in values}
        self._check_execution = check_execution
        self._authorize_entries = authorize_entries
        self._commit_exposure = commit_exposure
        self._read_content = read_content
        self._lock = asyncio.Lock()
        self._next_run_step = 1
        self._ticket: _ExposureTicket | None = None

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        projected, _ticket = await self._prepare_call(messages, generation=False)
        return await self.wrapped.count_tokens(
            projected,
            model_settings,
            model_request_parameters,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        projected, ticket = await self._prepare_call(messages, generation=True)
        response = await self.wrapped.request(
            projected,
            model_settings,
            model_request_parameters,
        )
        await self._finish_generation(ticket, response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ):
        projected, ticket = await self._prepare_call(messages, generation=True)
        async with self.wrapped.request_stream(
            projected,
            model_settings,
            model_request_parameters,
            run_context,
        ) as response_stream:
            yield response_stream
        await self._finish_generation(ticket, response_stream.get())

    async def _prepare_call(
        self,
        messages: list[ModelMessage],
        *,
        generation: bool,
    ) -> tuple[list[ModelMessage], _ExposureTicket]:
        selected = self._select_entries(messages)
        async with self._lock:
            ticket = self._ticket
            if ticket is None:
                ticket = _ExposureTicket(self._next_run_step, selected)
                self._ticket = ticket
            elif ticket.suspended:
                selected_ids = {value.activation_id for value in selected}
                ticket_by_id = {value.activation_id: value for value in ticket.entries}
                if any(
                    activation_id not in ticket_by_id
                    or ticket_by_id[activation_id] != self._activations[activation_id]
                    for activation_id in selected_ids
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif selected != ticket.entries:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

            await self._check_execution(ticket.run_step)
            await self._authorize_entries(ticket.run_step, selected)
            if ticket.exposure is None and ticket.entries:
                exposure = await self._commit_exposure(ticket.run_step, ticket.entries)
                self._validate_exposure(ticket, exposure)
                ticket.exposure = exposure
            if ticket.entries and ticket.bodies is None:
                ticket.bodies = await self._read_bodies(ticket.entries)
            if generation:
                await self._check_execution(ticket.run_step)
                await self._authorize_entries(ticket.run_step, selected)
            return self._project(messages, ticket, selected), ticket

    async def _finish_generation(
        self,
        ticket: _ExposureTicket,
        response: ModelResponse,
    ) -> None:
        async with self._lock:
            if self._ticket is not ticket:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if response.state == "suspended":
                ticket.suspended = True
                return
            if response.state != "complete":
                return
            self._next_run_step = ticket.run_step + 1
            self._ticket = None

    def _select_entries(
        self,
        messages: Sequence[ModelMessage],
    ) -> tuple[ModelExposureEntry, ...]:
        selected: list[ModelExposureEntry] = []
        seen: set[str] = set()
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                    continue
                for item in part.content:
                    activation_id = _marker_activation_id(item)
                    if activation_id is None or activation_id in seen:
                        continue
                    entry = self._activations.get(activation_id)
                    if entry is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    selected.append(entry)
                    seen.add(activation_id)
        return tuple(selected)

    async def _read_bodies(
        self,
        entries: Sequence[ModelExposureEntry],
    ) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for value in entries:
            body = await self._read_content(value.entry.content)
            reference = value.entry.content.object
            if (
                len(body) != reference.size
                or hashlib.sha256(body).hexdigest() != reference.digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[value.activation_id] = body
        return result

    def _project(
        self,
        messages: list[ModelMessage],
        ticket: _ExposureTicket,
        selected: tuple[ModelExposureEntry, ...],
    ) -> list[ModelMessage]:
        if not selected:
            return list(messages)
        if ticket.exposure is None or ticket.bodies is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        selected_ids = {value.activation_id for value in selected}
        expanded: set[str] = set()
        projected: list[ModelMessage] = []
        last_request_index = max(
            (index for index, value in enumerate(messages) if isinstance(value, ModelRequest)),
            default=-1,
        )
        for message_index, message in enumerate(messages):
            if not isinstance(message, ModelRequest):
                projected.append(message)
                continue
            parts = []
            changed = False
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                    parts.append(part)
                    continue
                content: list[UserContent] = []
                part_changed = False
                for item in part.content:
                    activation_id = _marker_activation_id(item)
                    if (
                        activation_id is None
                        or activation_id not in selected_ids
                        or activation_id in expanded
                    ):
                        content.append(item)
                        continue
                    activation = self._activations[activation_id]
                    content.append(
                        BinaryContent(
                            ticket.bodies[activation_id],
                            media_type=activation.entry.media_type,
                            identifier=activation.entry.presentation.identifier,
                            vendor_metadata=(
                                None
                                if activation.entry.presentation.vendor_metadata is None
                                else dict(activation.entry.presentation.vendor_metadata)
                            ),
                        )
                    )
                    expanded.add(activation_id)
                    part_changed = True
                if part_changed:
                    parts.append(replace(part, content=tuple(content)))
                    changed = True
                else:
                    parts.append(part)
            metadata = message.metadata
            if message_index == last_request_index:
                metadata = dict(metadata or {})
                current = metadata.get(_EXPOSURE_METADATA_KEY)
                if current not in {None, ticket.exposure.exposure_id}:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                metadata[_EXPOSURE_METADATA_KEY] = ticket.exposure.exposure_id
                changed = True
            projected.append(
                replace(message, parts=tuple(parts), metadata=metadata)
                if changed
                else message
            )
        if expanded != selected_ids:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return projected

    def _validate_exposure(
        self,
        ticket: _ExposureTicket,
        exposure: ModelExposure,
    ) -> None:
        if (
            not isinstance(exposure, ModelExposure)
            or exposure.execution_id != self._execution_id
            or exposure.step_run_id != self._step_run_id
            or exposure.run_step != ticket.run_step
            or exposure.path_origin != self._path_origin
            or exposure.entries != ticket.entries
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def attachment_placeholder(value: ModelExposureEntry) -> TextContent:
    """Build the only trusted in-message marker understood by AttachmentModel."""
    if not isinstance(value, ModelExposureEntry):
        raise TypeError("value must be ModelExposureEntry")
    label = value.entry.name or value.entry.path
    return TextContent(
        f"[attachment: {label}]",
        metadata={_MARKER_KEY: value.activation_id},
    )


def _marker_activation_id(value: UserContent) -> str | None:
    if not isinstance(value, TextContent):
        return None
    metadata = value.metadata
    if not isinstance(metadata, Mapping) or _MARKER_KEY not in metadata:
        return None
    if set(metadata) != {_MARKER_KEY}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    activation_id = metadata[_MARKER_KEY]
    if (
        not isinstance(activation_id, str)
        or len(activation_id) != 64
        or any(character not in "0123456789abcdef" for character in activation_id)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return activation_id


__all__: tuple[str, ...] = ()
