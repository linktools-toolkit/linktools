#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction overlay and tool-boundary coordination."""

import asyncio
from typing import TYPE_CHECKING

from ..capability import ToolCallRetry
from ..errors import AIError, ErrorCode
from ..workspace import RepositoryInstructions
from .state._contracts import ExecutionRecord, RecoveryCheckpoint

if TYPE_CHECKING:
    from ._recovery_coordinator import _RecoveryCoordinator

_REPOSITORY_INSTRUCTION_RECONSIDER = (
    "Repository instructions applicable to the target changed. Re-check the current "
    "instructions before retrying the call."
)


def _merge_repository_instructions(
    initial: RepositoryInstructions | None,
    overlay: RepositoryInstructions | None,
) -> RepositoryInstructions | None:
    if initial is None:
        return overlay
    if overlay is None:
        return initial
    initial_sources = {document.source for document in initial.documents}
    if any(document.source in initial_sources for document in overlay.documents):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return RepositoryInstructions((*initial.documents, *overlay.documents))


def _repository_instruction_signature(
    value: RepositoryInstructions | None,
) -> tuple[tuple[str, str, str], ...]:
    if value is None:
        return ()
    return tuple(
        (document.source, document.scope, document.content)
        for document in value.documents
    )


def _repository_instructions_contain(
    current: RepositoryInstructions | None,
    expected: RepositoryInstructions | None,
) -> bool:
    expected_documents = {
        document.source: (document.scope, document.content)
        for document in (() if expected is None else expected.documents)
    }
    if not expected_documents:
        return True
    current_documents = {
        document.source: (document.scope, document.content)
        for document in (() if current is None else current.documents)
    }
    return all(
        current_documents.get(source) == expected_value
        for source, expected_value in expected_documents.items()
    )


def _validate_repository_instruction_frontier(
    checkpoint: RecoveryCheckpoint,
    overlay: RepositoryInstructions | None,
) -> None:
    reference = checkpoint.repository_instruction_overlay
    barriers = checkpoint.repository_instruction_barriers
    if reference is None:
        if overlay is not None or barriers:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return
    if overlay is None or not barriers:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if (
        reference.payload.digest != overlay.digest
        or barriers[-1].resulting_overlay_digest != overlay.digest
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


class _RepositoryInstructionBoundary:
    def __init__(
        self,
        runtime: "_RecoveryCoordinator",
        execution: ExecutionRecord,
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
    ) -> None:
        self._runtime = runtime
        self._execution = execution
        self._initial = initial
        self._overlay = overlay
        self._has_initial = bool(initial is not None and initial.documents)
        self._initial_text = "" if not self._has_initial else initial.render()
        self._overlay_text = self._render_overlay_text(overlay)
        self._lock = asyncio.Lock()
        self._revision = 0
        self._observed_revision = -1

    def _render_overlay_text(
        self, overlay: RepositoryInstructions | None
    ) -> str:
        if overlay is None or not overlay.documents:
            return ""
        return overlay.render(include_preamble=not self._has_initial)

    def render_initial(self) -> str:
        return self._initial_text

    def render_overlay(self) -> str:
        self._observed_revision = self._revision
        return self._overlay_text

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> None:
        async with self._lock:
            if self._observed_revision != self._revision:
                raise ToolCallRetry(_REPOSITORY_INSTRUCTION_RECONSIDER)
            previous_signature = _repository_instruction_signature(self._overlay)
            next_overlay, reconsider = (
                await self._runtime.check_repository_instructions(
                    execution=self._execution,
                    initial=self._initial,
                    overlay=self._overlay,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    path_fields=path_fields,
                )
            )
            next_signature = _repository_instruction_signature(next_overlay)
            if next_signature != previous_signature:
                next_text = self._render_overlay_text(next_overlay)
                self._overlay = next_overlay
                self._overlay_text = next_text
                self._revision += 1
            else:
                self._overlay = next_overlay
            if reconsider:
                raise ToolCallRetry(_REPOSITORY_INSTRUCTION_RECONSIDER)
