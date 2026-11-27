#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction overlay and tool-boundary coordination."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import InstructionPart

from ..capability import AgentContext
from ..capability import ToolCallRetry
from ..core import PersistenceDomain
from ..errors import AIError, ErrorCode
from ..spec import RepositoryInstructionDocument, RepositoryInstructionResolver, RepositoryInstructions
from ._tool_boundary import RepositoryInstructionBoundary
from ..inputs import RepositoryInstructionPayload
from ..execution import ExecutionRecord

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
    *,
    fits_overlay: Callable[[RepositoryInstructions], bool] | None = None,
) -> RepositoryInstructions | None:
    active = _merge_repository_instructions(initial, overlay) or RepositoryInstructions()
    sources = {document.source for document in active.documents}
    discovered: list[RepositoryInstructionDocument] = []
    previous = () if overlay is None else overlay.documents
    for path in paths:
        resolved = await resolver.resolve(path, exclude_sources=frozenset(sources), within=active)
        added = []
        for document in resolved.documents:
            if document.source in sources:
                continue
            if fits_overlay is not None and not fits_overlay(RepositoryInstructions((*previous, *discovered, document))):
                continue
            sources.add(document.source)
            discovered.append(document)
            added.append(document)
        if added:
            active = RepositoryInstructions((*active.documents, *added))
    if not discovered:
        return None
    return RepositoryInstructions((*previous, *discovered))


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


def _validate_repository_instruction_overlay(
    execution: ExecutionRecord,
    overlay: RepositoryInstructions | None,
) -> None:
    if execution.recovery is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reference = execution.recovery.repository_instruction_overlay
    if reference is None:
        if overlay is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return
    if overlay is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    RepositoryInstructionPayload.from_reference(reference, source_domain=PersistenceDomain.CHECKPOINT)


class _RepositoryInstructionBoundary:
    def __init__(
        self,
        check_frontier: Callable[..., Awaitable[RepositoryInstructions | None]],
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
    ) -> None:
        self._check_frontier = check_frontier
        self._initial = initial
        self._overlay = overlay
        self._has_initial = bool(initial is not None and initial.documents)
        self._initial_text = "" if not self._has_initial else initial.render()
        self._overlay_text = self._render_overlay_text(overlay)
        self._lock = asyncio.Lock()

    def _render_overlay_text(
        self, overlay: RepositoryInstructions | None
    ) -> str:
        if overlay is None or not overlay.documents:
            return ""
        return overlay.render(include_preamble=not self._has_initial)

    def current_instructions(self) -> RepositoryInstructions | None:
        return _merge_repository_instructions(self._initial, self._overlay)

    def render_initial(self) -> str:
        return self._initial_text

    def render_overlay(self) -> str:
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
            try:
                next_overlay = (
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
            if next_overlay != self._overlay:
                self._overlay = next_overlay
                self._overlay_text = self._render_overlay_text(next_overlay)


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
