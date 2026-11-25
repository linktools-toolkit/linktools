#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction overlay and tool-boundary coordination."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import InstructionPart, ToolCallPart

from ..capability import AgentContext
from ..capability import ToolCallFailed, ToolCallRetry
from ..core import PersistenceDomain, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import RepositoryInstructionDocument, RepositoryInstructionResolver, RepositoryInstructions
from ._tool_boundary import RepositoryInstructionBoundary, is_instruction_file_read
from ..inputs import RepositoryInstructionPayload
from ..execution import ExecutionRecord

_REPOSITORY_INSTRUCTION_RECONSIDER = (
    "This call was made before the current repository instructions were available. "
    "Re-check the current instructions before retrying the call."
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


def _instruction_paths(
    arguments: Mapping[str, object],
    path_fields: tuple[str, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    for field in path_fields:
        value = arguments.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            paths.append(value)
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            paths.extend(value)
            continue
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return tuple(paths)

async def _discover_repository_overlay(
    resolver: RepositoryInstructionResolver,
    initial: RepositoryInstructions | None,
    overlay: RepositoryInstructions | None,
    paths: tuple[str, ...],
) -> RepositoryInstructions | None:
    active = _merge_repository_instructions(initial, overlay)
    sources = {document.source for document in (() if active is None else active.documents)}
    discovered: list[RepositoryInstructionDocument] = []
    for path in paths:
        resolved = await resolver.resolve(path, exclude_sources=frozenset(sources))
        for document in resolved.documents:
            if document.source not in sources:
                sources.add(document.source)
                discovered.append(document)
    if not discovered:
        return None
    next_overlay = _merge_repository_instructions(overlay, RepositoryInstructions(tuple(discovered)))
    assert next_overlay is not None
    resolver.validate(_merge_repository_instructions(initial, next_overlay))
    return next_overlay


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
    execution: ExecutionRecord,
    overlay: RepositoryInstructions | None,
) -> None:
    if execution.recovery is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reference = execution.recovery.repository_instruction_overlay
    barriers = execution.recovery.repository_instruction_barriers
    if reference is None:
        if overlay is not None or barriers:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return
    if overlay is None or not barriers:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    RepositoryInstructionPayload.from_reference(reference, source_domain=PersistenceDomain.CHECKPOINT)


@dataclass(frozen=True, slots=True)
class _RepositoryInstructionApproval:
    tool_name: str
    frontier_digest: str


class _RepositoryInstructionBoundary:
    def __init__(
        self,
        check_frontier: Callable[..., Awaitable[tuple[RepositoryInstructions | None, bool]]],
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
        approved_frontiers: Mapping[str, _RepositoryInstructionApproval] | None = None,
    ) -> None:
        self._check_frontier = check_frontier
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

    def approval_frontiers(self, calls: Sequence[ToolCallPart], *, omit_empty: bool = False) -> dict[str, str]:
        observed = self._observed_frontier
        if (omit_empty and observed is not None and not self._has_initial
                and (self._overlay is None or not self._overlay.documents)
                and observed == self._frontier_digest()):
            # An empty observation is not a repository instruction. Only the
            # split profile omits it; an earlier nonempty or checked frontier
            # must still reach approval persistence and its policy checks.
            observed = None
        result = {}
        for call in calls:
            if observed is not None:
                result[call.tool_call_id] = observed
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
            # Reading the guidance is how the model learns it; other calls must
            # wait for a request that has observed the newly pinned frontier.
            if self._observed_revision != self._revision and (
                approved or not is_instruction_file_read(tool_name, arguments, path_fields)
            ):
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
            try:
                next_overlay, reconsider = (
                    await self._check_frontier(
                        initial=self._initial,
                        overlay=self._overlay,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        arguments=arguments,
                        path_fields=path_fields,
                    )
                )
            except AIError as error:
                if (error.code is ErrorCode.STORAGE_PATH_INVALID
                        and error.safe_details.get("reason") == "repository_instruction_target_path_too_long"):
                    raise ToolCallRetry(
                        "The target path is too long. Use the exact path returned by a file tool "
                        "or request a valid path before retrying."
                    ) from error
                raise
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
