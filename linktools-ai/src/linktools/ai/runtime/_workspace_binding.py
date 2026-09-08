#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bind trusted model tool paths before runtime responses become durable."""

from collections.abc import Mapping, Sequence
from typing import Any

from linktools.core import environ
from pydantic_ai import AgentRunResult
from pydantic_ai.messages import ModelMessage, ToolCallPart

from ..capability import WorkspaceAccess
from ..core import canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from ..workspace import normalize_workspace_path
from .state import WorkspacePathBinding, WorkspaceToolCallBinding
from .state import WorkspaceToolCallBindingStore

_logger = environ.get_logger("ai.runtime.workspace_binding")


class WorkspaceToolCallBinder:
    """Canonicalize and durably bind trusted paths from one model response."""

    def __init__(
        self,
        store: WorkspaceToolCallBindingStore,
        access: WorkspaceAccess,
    ) -> None:
        self._store = store
        self._access = access

    async def bind_result(
        self,
        result: AgentRunResult[object],
        *,
        execution_id: str,
        step_run_id: str,
        path_fields: Mapping[str, Sequence[str]],
    ) -> None:
        await self.bind_messages(
            result.new_messages(),
            execution_id=execution_id,
            step_run_id=step_run_id,
            path_fields=path_fields,
        )

    async def bind_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        execution_id: str,
        step_run_id: str,
        path_fields: Mapping[str, Sequence[str]],
    ) -> None:
        if not execution_id or not step_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not path_fields:
            return
        for message in messages:
            if message.run_id != step_run_id:
                continue
            for part in message.parts:
                if not isinstance(part, ToolCallPart):
                    continue
                fields = path_fields.get(part.tool_name)
                if fields is None:
                    continue
                existing = await self._store.get(
                    step_run_id,
                    part.tool_call_id,
                )
                if existing is not None:
                    self._validate_existing(
                        existing,
                        part,
                        execution_id=execution_id,
                        step_run_id=step_run_id,
                    )
                    continue
                binding = await self._bind_call(
                    part,
                    fields=fields,
                    execution_id=execution_id,
                    step_run_id=step_run_id,
                )
                await self._store.store(binding)
                _logger.info(
                    "workspace tool call binding committed: execution=%s step=%s "
                    "tool=%s call=%s paths=%s error=%s",
                    execution_id,
                    step_run_id,
                    part.tool_name,
                    part.tool_call_id,
                    len(binding.paths),
                    binding.error_code,
                )

    async def validate_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        execution_id: str,
        step_run_id: str,
        path_fields: Mapping[str, Sequence[str]],
    ) -> None:
        """Verify bindings for durable messages without reinterpreting paths."""
        if not execution_id or not step_run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not path_fields:
            return
        for message in messages:
            if message.run_id != step_run_id:
                continue
            for part in message.parts:
                if not isinstance(part, ToolCallPart):
                    continue
                if part.tool_name not in path_fields:
                    continue
                binding = await self._store.get(
                    step_run_id,
                    part.tool_call_id,
                )
                if binding is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._validate_existing(
                    binding,
                    part,
                    execution_id=execution_id,
                    step_run_id=step_run_id,
                )

    @staticmethod
    def _validate_existing(
        binding: WorkspaceToolCallBinding,
        call: ToolCallPart,
        *,
        execution_id: str,
        step_run_id: str,
    ) -> None:
        try:
            arguments_digest = canonical_sha256(
                normalize_json_value(call.args_as_dict())
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (
            binding.execution_id != execution_id
            or binding.step_run_id != step_run_id
            or binding.tool_call_id != call.tool_call_id
            or binding.tool_name != call.tool_name
            or binding.arguments_digest != arguments_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _bind_call(
        self,
        call: ToolCallPart,
        *,
        fields: Sequence[str],
        execution_id: str,
        step_run_id: str,
    ) -> WorkspaceToolCallBinding:
        try:
            arguments = call.args_as_dict()
            normalized_arguments = normalize_json_value(arguments)
            arguments_digest = canonical_sha256(normalized_arguments)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        try:
            paths = await self._canonicalize_arguments(
                arguments,
                fields=fields,
            )
        except AIError as error:
            return WorkspaceToolCallBinding(
                1,
                execution_id,
                step_run_id,
                call.tool_call_id,
                call.tool_name,
                arguments_digest,
                (),
                error.code.value,
            )
        return WorkspaceToolCallBinding(
            1,
            execution_id,
            step_run_id,
            call.tool_call_id,
            call.tool_name,
            arguments_digest,
            tuple(paths),
            None,
        )

    async def _canonicalize_arguments(
        self,
        arguments: Mapping[str, Any],
        *,
        fields: Sequence[str],
    ) -> tuple[WorkspacePathBinding, ...]:
        result: list[WorkspacePathBinding] = []
        for field in fields:
            if field not in arguments:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            value = arguments[field]
            if isinstance(value, str):
                values = (value,)
            elif isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                values = tuple(value)
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            for index, path in enumerate(values):
                if not isinstance(path, str) or not path:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                try:
                    canonical = normalize_workspace_path(
                        await self._access.canonicalize_path(path)
                    )
                except (AIError, TypeError, ValueError) as error:
                    if isinstance(error, AIError):
                        raise
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
                pointer = _json_pointer(field)
                if not isinstance(value, str):
                    pointer += f"/{index}"
                result.append(WorkspacePathBinding(pointer, canonical))
        return tuple(result)


def _json_pointer(field: str) -> str:
    return "/" + field.replace("~", "~0").replace("/", "~1")


__all__ = ["WorkspaceToolCallBinder"]
