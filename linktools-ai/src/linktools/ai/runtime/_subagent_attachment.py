#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned child input preparation from parent-authorized attachment grants."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from ..core import Principal
from ..errors import AIError, ErrorCode
from ._attachment import _path_origin, _stable_prepare_error
from ._input import input_intent_digest
from .state import (
    AttachmentEntry,
    ExecutionRecord,
    InputAttachmentPart,
    InputPrepareRecord,
    InputPrepareSlot,
    InputTextPart,
    InputV2,
    PreparedInput,
    input_v2_digest,
    managed_attachment_path,
)
from .state._attachment_repository import AttachmentRepository

_Grant = Callable[[str], Awaitable[AttachmentEntry]]


class SubagentAttachmentPreparer:
    """Freeze one explicit parent grant set into a direct child PreparedInput."""

    def __init__(
        self,
        repository: AttachmentRepository,
        workspace,
    ) -> None:
        if not isinstance(repository, AttachmentRepository):
            raise TypeError("repository must be AttachmentRepository")
        self._repository = repository
        self._workspace = workspace

    async def adopted(
        self,
        task: str,
        attachments: Sequence[str],
        *,
        idempotency_key: str,
    ) -> bool:
        paths = _paths(attachments)
        owner_key = self._repository.prepare_key(
            "execution.subagent",
            idempotency_key,
        )
        current = await self._repository.get_prepare(
            owner_key,
            tenant_id=self._repository._tenant_id,
        )
        if current is None:
            return False
        if current.intent_digest != input_intent_digest(task, paths):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if current.status == "ADOPTED":
            if current.target is None or current.input is not None or current.slots:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return True
        if current.status == "ABORTED":
            raise _stable_prepare_error(current.error_code)
        if current.status not in {"PREPARING", "READY"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return False

    async def prepare(
        self,
        task: str,
        attachments: Sequence[str],
        *,
        principal: Principal,
        idempotency_key: str,
        grant: _Grant,
    ) -> PreparedInput:
        paths = _paths(attachments)
        intent_digest = input_intent_digest(task, paths)
        origin = _path_origin(self._workspace)
        candidate = InputPrepareRecord(
            1,
            intent_digest,
            origin,
            "PREPARING",
            (),
            None,
            None,
            None,
        )
        owner_key, current = await self._repository.reserve_prepare(
            "execution.subagent",
            idempotency_key,
            candidate,
        )
        if current.status == "READY":
            if current.input is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current.input
        if current.status == "ADOPTED":
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.status == "ABORTED":
            raise _stable_prepare_error(current.error_code)
        if current.status != "PREPARING":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        unique_paths = tuple(dict.fromkeys(paths))
        for slot, path in enumerate(unique_paths):
            if slot < len(current.slots):
                frozen = current.slots[slot]
                if (
                    frozen.slot != slot
                    or frozen.entry.path
                    != managed_attachment_path("p", owner_key, slot)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                continue
            if slot != len(current.slots):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source = await grant(path)
            entry = AttachmentEntry(
                managed_attachment_path("p", owner_key, slot),
                source.name,
                source.media_type,
                source.presentation,
                source.content,
            )
            next_record = replace(
                current,
                slots=(
                    *current.slots,
                    InputPrepareSlot(slot, None, entry),
                ),
            )
            current = await self._repository.compare_and_swap_prepare(
                owner_key,
                expected=current,
                next_record=next_record,
            )

        manifest = tuple(item.entry for item in current.slots)
        prepared = self._prepared(
            task,
            paths,
            manifest,
            intent_digest=intent_digest,
            input_digest=None,
            origin=origin,
        )
        ready = InputPrepareRecord(
            1,
            intent_digest,
            origin,
            "READY",
            (),
            prepared,
            None,
            None,
        )
        committed = await self._repository.compare_and_swap_prepare(
            owner_key,
            expected=current,
            next_record=ready,
        )
        if committed.input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return committed.input

    def replay(
        self,
        task: str,
        attachments: Sequence[str],
        execution: ExecutionRecord,
    ) -> PreparedInput:
        """Rebuild the child lightweight input from its adopted Execution truth."""
        paths = _paths(attachments)
        if (
            not execution.attachment_manifest
            or execution.input_digest is None
            or execution.path_origin is None
        ):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return self._prepared(
            task,
            paths,
            execution.attachment_manifest,
            intent_digest=input_intent_digest(task, paths),
            input_digest=execution.input_digest,
            origin=execution.path_origin,
        )

    @staticmethod
    def _prepared(
        task: str,
        paths: tuple[str, ...],
        manifest: tuple[AttachmentEntry, ...],
        *,
        intent_digest: str,
        input_digest: str | None,
        origin,
    ) -> PreparedInput:
        unique_paths = tuple(dict.fromkeys(paths))
        if len(manifest) != len(unique_paths):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        by_path = {path: index for index, path in enumerate(unique_paths)}
        prompt = InputV2(
            2,
            (
                InputTextPart("text", task),
                *(InputAttachmentPart("attachment", by_path[path]) for path in paths),
            ),
            (),
            (),
        )
        calculated = input_v2_digest(prompt, manifest)
        if input_digest is not None and input_digest != calculated:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return PreparedInput(
            1,
            "linktools-input-v2",
            prompt,
            manifest,
            intent_digest,
            calculated,
            origin,
        )


def _paths(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result = tuple(value)
    if not result or any(not isinstance(path, str) or not path for path in result):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return result


__all__ = ["SubagentAttachmentPreparer"]
