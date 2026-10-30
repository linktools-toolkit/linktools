#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction overlay and tool-boundary coordination."""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import InstructionPart, ToolCallPart

from ..capability import AgentContext
from ..capability import ToolCallFailed
from ..core import canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import RepositoryInstructions
from ._tool_boundary import RepositoryInstructionBoundary
from ..execution import ExecutionRecord
from ..persistence.contracts import RecoveryCheckpoint

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
    overlay_digest = canonical_sha256(overlay.to_payload())
    if reference.payload.digest != overlay_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


@dataclass(frozen=True, slots=True)
class _RepositoryInstructionApproval:
    tool_name: str
    frontier_digest: str


class _RepositoryInstructionBoundary:
    def __init__(
        self,
        runtime: "_RecoveryCoordinator",
        execution: ExecutionRecord,
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
        producer_generation: int | None = None,
        approved_frontiers: Mapping[str, _RepositoryInstructionApproval] | None = None,
    ) -> None:
        self._runtime = runtime
        self._execution = execution
        self._producer_generation = producer_generation
        self._initial = initial
        self._overlay = overlay
        self._has_initial = bool(initial is not None and initial.documents)
        self._initial_text = "" if not self._has_initial else initial.render()
        self._overlay_text = self._render_overlay_text(overlay)
        self._lock = asyncio.Lock()
        self._revision = 0
        self._observed_revision = -1
        self._approved_frontiers = dict(approved_frontiers or {})
        self._checked_approvals: dict[str, _RepositoryInstructionApproval] = {}
        self._observed_frontier: str | None = None

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
        self._observed_frontier = self._frontier_digest()
        return self._overlay_text

    def _frontier_digest(self) -> str:
        return canonical_sha256({
            "initial": [list(document) for document in _repository_instruction_signature(self._initial)],
            "overlay": [list(document) for document in _repository_instruction_signature(self._overlay)],
        })

    def approval_frontiers(self, calls: Sequence[ToolCallPart]) -> dict[str, str]:
        result = {}
        for call in calls:
            if self._observed_frontier is not None:
                result[call.tool_call_id] = self._observed_frontier
            else:
                # Deferred replay may pause again without a new model request.
                # Only that original approved call can carry its observed frontier.
                checked = self._checked_approvals.get(call.tool_call_id)
                if checked is not None and checked.tool_name == call.tool_name:
                    result[call.tool_call_id] = checked.frontier_digest
        return result

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
        approved: bool = False,
    ) -> None:
        async with self._lock:
            if self._observed_revision != self._revision:
                permit = self._approved_frontiers.get(tool_call_id)
                if (
                    not approved
                    or permit is None
                    or permit.tool_name != tool_name
                    or permit.frontier_digest != self._frontier_digest()
                ):
                    raise ToolCallFailed(_REPOSITORY_INSTRUCTION_RECONSIDER)
                # The SDK binds approval to the frozen original args through
                # ToolApproved.override_args; its validators may normalize them.
                self._checked_approvals[tool_call_id] = permit
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
                    producer_generation=self._producer_generation,
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
                raise ToolCallFailed(_REPOSITORY_INSTRUCTION_RECONSIDER)


class _RepositoryInstructionCapability(Capability[AgentContext[object]]):
    """Inject the persisted instruction set and its execution overlay once."""

    def __init__(
        self,
        initial: RepositoryInstructions | None,
        boundary: RepositoryInstructionBoundary | None,
    ) -> None:
        initial_text = (
            boundary.render_initial()
            if boundary is not None
            else "" if initial is None else initial.render()
        )
        instructions = (
            ()
            if not initial_text
            else (InstructionPart(content=initial_text, dynamic=False),)
        )
        super().__init__(
            id="linktools.ai.repository-instructions",
            instructions=instructions,
        )
        self._boundary = boundary
        if boundary is not None:
            self.instructions(self._render_overlay)

    def _render_overlay(
        self,
        _context: RunContext[AgentContext[object]],
    ) -> str | None:
        if self._boundary is None:
            return None
        value = self._boundary.render_overlay()
        return value or None
