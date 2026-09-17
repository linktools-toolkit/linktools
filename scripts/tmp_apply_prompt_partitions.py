#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary patcher for the prompt partition implementation."""

from pathlib import Path


PATH = Path("linktools-ai/src/linktools/ai/runtime/_local.py")


def replace_between(
    value: str,
    start: str,
    end: str,
    replacement: str,
    *,
    offset: int = 0,
) -> str:
    start_index = value.index(start, offset)
    end_index = value.index(end, start_index)
    return value[:start_index] + replacement + value[end_index:]


def replace_once(value: str, old: str, new: str, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"unexpected {label} count: {count}")
    return value.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text()

    boundary_replacement = '''def _repository_instruction_signature(
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


'''
    text = replace_between(
        text,
        "class _RepositoryInstructionBoundary:\n",
        "_REPOSITORY_INSTRUCTION_RECONSIDER = (\n",
        boundary_replacement,
    )

    coordinator_class = text.index("class _RecoveryCoordinator:\n")
    coordinator_replacement = '''    async def check_repository_instructions(
        self,
        *,
        execution: ExecutionRecord,
        initial: RepositoryInstructions | None,
        overlay: RepositoryInstructions | None,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> tuple[RepositoryInstructions | None, bool]:
        del tool_name
        paths = _instruction_paths(arguments, path_fields)
        resolver = self._instruction_resolver
        if not paths or resolver is None:
            return overlay, False
        checkpoint = await self._port.load_recovery_checkpoint(
            execution.execution_id,
            tenant_id=execution.tenant_id,
        )
        if (
            checkpoint is None
            or checkpoint.state is not RecoveryCheckpointState.ACTIVE
            or checkpoint.step_run_id is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        persisted_overlay = await self._port.load_repository_instructions(
            checkpoint.repository_instruction_overlay
        )
        _validate_repository_instruction_frontier(checkpoint, persisted_overlay)
        arguments_digest = canonical_sha256(normalize_json_value(arguments))
        matching = tuple(
            barrier
            for barrier in checkpoint.repository_instruction_barriers
            if barrier.step_run_id == checkpoint.step_run_id
            and barrier.tool_call_id == tool_call_id
        )
        if matching:
            barrier = matching[0]
            if barrier.arguments_digest != arguments_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return persisted_overlay, True

        active = _merge_repository_instructions(initial, persisted_overlay)
        excluded = frozenset(
            () if active is None else document.source for document in active.documents
        )
        discovered: list[object] = []
        discovered_sources = set(excluded)
        for path in paths:
            resolved = await resolver.resolve(
                path,
                exclude_sources=frozenset(discovered_sources),
            )
            for document in resolved.documents:
                if document.source in discovered_sources:
                    continue
                discovered_sources.add(document.source)
                discovered.append(document)
        if not discovered:
            return persisted_overlay, (
                _repository_instruction_signature(overlay)
                != _repository_instruction_signature(persisted_overlay)
            )
        new_documents = RepositoryInstructions(tuple(discovered))
        next_overlay = _merge_repository_instructions(persisted_overlay, new_documents)
        if next_overlay is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        barrier = RepositoryInstructionBarrier(
            checkpoint.step_run_id,
            tool_call_id,
            arguments_digest,
            next_overlay.digest,
        )
        committed = await self._port.commit_repository_instruction_barrier(
            execution,
            checkpoint,
            next_overlay,
            barrier,
        )
        committed_overlay = await self._port.load_repository_instructions(
            committed.repository_instruction_overlay
        )
        _validate_repository_instruction_frontier(committed, committed_overlay)
        if (
            committed_overlay is None
            or not _repository_instructions_contain(committed_overlay, next_overlay)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "repository instructions extended: execution=%s step=%s tool_call=%s",
            execution.execution_id,
            checkpoint.step_run_id,
            tool_call_id,
        )
        return committed_overlay, True

'''
    text = replace_between(
        text,
        "    async def check_repository_instructions(\n",
        "    async def commit_deferred_pause(\n",
        coordinator_replacement,
        offset=coordinator_class,
    )

    store_index = text.index("    async def _store_repository_instructions(\n")
    commit_replacement = '''    async def commit_repository_instruction_barrier(
        self,
        execution: ExecutionRecord,
        checkpoint: RecoveryCheckpoint,
        overlay: RepositoryInstructions,
        barrier: RepositoryInstructionBarrier,
    ) -> RecoveryCheckpoint:
        if barrier.resulting_overlay_digest != overlay.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = tuple(
            item
            for item in checkpoint.repository_instruction_barriers
            if item.step_run_id == barrier.step_run_id
            and item.tool_call_id == barrier.tool_call_id
        )
        if existing:
            if existing[0] != barrier:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            current = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current_overlay = await self.load_repository_instructions(
                current.repository_instruction_overlay
            )
            _validate_repository_instruction_frontier(current, current_overlay)
            if (
                current_overlay is None
                or not _repository_instructions_contain(current_overlay, overlay)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        overlay_reference = await self._store_repository_instructions(
            execution,
            overlay,
        )
        next_checkpoint = replace(
            checkpoint,
            repository_instruction_overlay=overlay_reference,
            repository_instruction_barriers=(
                *checkpoint.repository_instruction_barriers,
                barrier,
            ),
            revision=checkpoint.revision + 1,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._recovery.checkpoints.compare_and_swap(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_revision=checkpoint.revision,
                next_record=next_checkpoint,
            )
        except AIError as error:
            current = await self._recovery.checkpoints.get(
                execution.execution_id,
                tenant_id=execution.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            matching = tuple(
                item
                for item in current.repository_instruction_barriers
                if item.step_run_id == barrier.step_run_id
                and item.tool_call_id == barrier.tool_call_id
            )
            if matching:
                if matching[0] != barrier:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
                current_overlay = await self.load_repository_instructions(
                    current.repository_instruction_overlay
                )
                _validate_repository_instruction_frontier(current, current_overlay)
                if (
                    current_overlay is None
                    or not _repository_instructions_contain(current_overlay, overlay)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                return current
            raise

'''
    text = replace_between(
        text,
        "    async def commit_repository_instruction_barrier(\n",
        "    async def _commit_failure(\n",
        commit_replacement,
        offset=store_index,
    )

    old_setup = '''            initial_repository_instructions = await self.load_repository_instructions(
                current.repository_instructions
            )
            repository_overlay = await self.load_repository_instructions(
                checkpoint.repository_instruction_overlay
            )
            repository_instructions = _merge_repository_instructions(
                initial_repository_instructions,
                repository_overlay,
            )
'''
    new_setup = '''            initial_repository_instructions = await self.load_repository_instructions(
                current.repository_instructions
            )
            repository_overlay = await self.load_repository_instructions(
                checkpoint.repository_instruction_overlay
            )
            _validate_repository_instruction_frontier(checkpoint, repository_overlay)
'''
    text = replace_once(text, old_setup, new_setup, "repository setup")

    old_boundary = '''            repository_boundary = _RepositoryInstructionBoundary(
                self._recovery_coordinator,
                current,
                initial_repository_instructions,
                repository_instructions,
            )
'''
    new_boundary = '''            repository_boundary = _RepositoryInstructionBoundary(
                self._recovery_coordinator,
                current,
                initial_repository_instructions,
                repository_overlay,
            )
'''
    text = replace_once(text, old_boundary, new_boundary, "boundary construction")

    text = replace_once(
        text,
        "                        repository_instructions=repository_instructions,\n",
        "                        repository_instructions=initial_repository_instructions,\n",
        "scope repository",
    )

    PATH.write_text(text)


if __name__ == "__main__":
    main()
