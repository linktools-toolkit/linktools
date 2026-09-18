#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime StepStore orchestration over durable archives."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from time import monotonic
from uuid import uuid4

from linktools.core import environ

from ...errors import AIError, ErrorCode
from ._contracts import (
    ExecutionRunSealHead,
    LoadedContextMessage,
    LoadedModelContext,
    ModelInteractionRecord,
    TranscriptMessageRef,
)
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._plan import RuntimeDomain, RuntimeRetentionMode
from ._step_archive import (
    CapturedExecutionProjection,
    ExecutionProjectionBatch,
    ExecutionTerminalSealPlan,
    InMemoryStepArchive,
    LockOrderError,
    PreparedExecutionProjection,
    PreparedStepSnapshot,
    PreparedStepSnapshotBatch,
    StagingStepStore,
    StateStepArchive,
    _LocalExecutionTerminalSeal,
    _ProjectionOffset,
    _RunDurabilityFlight,
    _RunDurabilityKind,
    _RunHistoryLock,
    _RunProjectionFlight,
    _StepArchiveBatch,
    _materialize_snapshot,
    _sync_projection,
)
from ._step_contracts import ContinuableSnapshot, RunRecord, StepEvent, StepStore

_logger = environ.get_logger("ai.runtime.state.steps")


class RuntimeStepStore(StepStore):
    """Route staging facts to their owning durable StateStore archive."""

    def __init__(
        self,
        staging: StagingStepStore,
        *,
        conversation_archive: StepStore,
        execution_archive: StepStore | None,
        recovery_archive: StepStore | None,
        conversation_retention: RuntimeRetentionMode,
        execution_retention: RuntimeRetentionMode,
        recovery_retention: RuntimeRetentionMode,
    ) -> None:
        del conversation_retention, execution_retention, recovery_retention
        self._staging = staging
        self._archives = {
            RuntimeDomain.CONVERSATION: conversation_archive,
            **({RuntimeDomain.EXECUTION: execution_archive} if execution_archive is not None else {}),
            **({RuntimeDomain.RECOVERY: recovery_archive} if recovery_archive is not None else {}),
        }
        self._initialized = False
        self._preflight = False
        self._projection_offsets: dict[str, _ProjectionOffset] = {}
        self._projection_dirty: set[str] = set()
        self._durability_flights: dict[str, _RunDurabilityFlight] = {}
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._terminal_seals: dict[str, _LocalExecutionTerminalSeal] = {}
        self._history_lock = _RunHistoryLock()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.bind_history_lock(self._history_lock)

    async def initialize(self) -> None:
        await self._staging.initialize()
        for archive in self._archives.values():
            await archive.initialize()
        self._projection_offsets.clear()
        self._projection_dirty.clear()
        self._durability_flights.clear()
        self._terminal_seals.clear()
        self._initialized = True

    async def validate_integrity(self) -> None:
        await self._ensure_business()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                await archive.validate_integrity()

    def register_context_baseline(
        self,
        step_run_id: str,
        context: LoadedModelContext,
    ) -> None:
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.register_context_baseline(step_run_id, context)

    async def register_run(
        self,
        record: RunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        async with self._history_lock.hold(record.run_id):
            self._ensure_run_mutable(record.run_id)
            self._staging.register_run_local(record)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        await self._ensure_business()
        return await self._staging.get_run(run_id=run_id)

    async def list_runs(
        self, *, parent_run_id: str | None = None, conversation_id: str | None = None
    ) -> list[RunRecord]:
        await self._ensure_business()
        return await self._staging.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def append_event(
        self,
        event: StepEvent,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        async with self._history_lock.hold(event.run_id):
            self._ensure_run_mutable(event.run_id)
            self._staging.append_event_local(event)
            self._projection_dirty.add(event.run_id)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self._ensure_business()
        return await self._staging.list_events(run_id=run_id)

    async def save_snapshot(
        self,
        snapshot: ContinuableSnapshot,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        while True:
            completion: asyncio.Future[None] | None = None
            recovery: StepStore | None = None
            recovery_run: RunRecord | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(snapshot.run_id):
                existing = self._durability_flights.get(snapshot.run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(snapshot.run_id)
                    self._staging.save_snapshot_local(snapshot)
                    self._projection_dirty.add(snapshot.run_id)
                    recovery = self._archives.get(RuntimeDomain.RECOVERY)
                    recovery_run = self._staging.get_run_local(snapshot.run_id)
                    if recovery is not None:
                        flight = self._install_durability_flight_locked(
                            snapshot.run_id,
                            _RunDurabilityKind.SNAPSHOT,
                        )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if recovery is None:
                return
            if recovery_run is None or flight is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)

            async def operation(
                target_recovery: StepStore = recovery,
                target_run: RunRecord = recovery_run,
                target_snapshot: ContinuableSnapshot = snapshot,
            ) -> None:
                await _materialize_snapshot(
                    target_recovery,
                    target_run,
                    target_snapshot,
                )

            async def readback(
                target_recovery: StepStore = recovery,
                target_run: RunRecord = recovery_run,
                target_snapshot: ContinuableSnapshot = snapshot,
            ) -> CommitObservation[None]:
                try:
                    observed_run = await target_recovery.get_run(
                        run_id=target_snapshot.run_id
                    )
                    observed_snapshot = await target_recovery.latest_snapshot(
                        run_id=target_snapshot.run_id,
                        include_interrupted=True,
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if observed_run == target_run and observed_snapshot == target_snapshot:
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        await self._ensure_business()
        return await self._staging.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)

    def intern_payload(self, run_id: str, payload: bytes) -> tuple[str, int]:
        return self._staging.intern_payload(run_id, payload)

    def staged_payload(self, run_id: str, digest: str) -> bytes:
        return self._staging.staged_payload(run_id, digest)

    def stage_model_interaction(self, interaction: object) -> None:
        self._staging.stage_model_interaction(interaction)
        run_id = getattr(interaction, "run_id", None)
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("staged model interaction has no run id")
        self._projection_dirty.add(run_id)

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        await self._ensure_business()
        return await self._staging.list_model_interactions(
            run_id=run_id,
            after_request_sequence=after_request_sequence,
            limit=limit,
        )

    async def resolve_model_interaction(self, interaction: object) -> object:
        await self._ensure_business()
        return await self._staging.resolve_model_interaction(interaction)

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        await self._ensure_business()
        return await self._staging.resolve_model_interactions(interactions)

    def read_store(self, runtime_domain: RuntimeDomain) -> StepStore:
        if runtime_domain not in self._archives:
            return self._staging
        return self._archives[runtime_domain]

    async def load_loaded_model_context(
        self,
        runtime_domain: RuntimeDomain,
        owner_id: str,
    ) -> LoadedModelContext:
        archive = self._archives.get(runtime_domain)
        if archive is None:
            archive = self._staging
        if isinstance(archive, (StateStepArchive, StagingStepStore)):
            if isinstance(archive, StateStepArchive):
                return await archive.load_loaded_model_context(
                    owner_id=owner_id,
                )
            return await archive.load_loaded_model_context(owner_id=owner_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def resolve_transcript_message_refs(
        self,
        refs: Sequence[TranscriptMessageRef],
    ) -> tuple[LoadedContextMessage, ...]:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        return await archive.resolve_transcript_message_refs(refs)

    async def iter_session_messages(
        self,
        history_id: str,
        *,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        async for message in archive.transcript_repository.iter_session_messages(
            history_id,
            tenant_id=tenant_id,
        ):
            yield message

    def iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_messages(
            history_id=history_id,
            step_run_id=step_run_id,
            tenant_id=tenant_id,
        )

    async def _iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            async for message in archive.iter_session_messages(
                history_id,
                tenant_id=tenant_id,
            ):
                yield message
            return
        if isinstance(archive, InMemoryStepArchive):
            async for message in archive.iter_messages(run_id=step_run_id):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def conversation_message_count(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
    ) -> int:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await archive.session_message_count(
                history_id,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            return await archive.transcript_message_count(step_run_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_message_range(
            history_id=history_id,
            step_run_id=step_run_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    async def _iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            async for message in archive.iter_session_message_range(
                history_id,
                tenant_id=tenant_id,
                start=start,
                end=end,
            ):
                yield message
            return
        if isinstance(archive, InMemoryStepArchive):
            async for message in archive.iter_message_range(
                run_id=step_run_id,
                start=start,
                end=end,
            ):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def load_conversation_model_context(
        self,
        *,
        history_id: str | None,
        step_run_id: str,
        tenant_id: str,
    ) -> tuple[object, ...]:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await archive.load_session_model_context(
                history_id,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            return await archive.load_model_context(run_id=step_run_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def materialize_recovery_snapshot(self, *, step_run_id: str, require_complete: bool) -> None:
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id, include_interrupted=True)
        run = await self._staging.get_run(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.RECOVERY)
        if snapshot is None or run is None:
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if require_complete and snapshot.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if archive is not None:
            await _materialize_snapshot(archive, run, snapshot)
            interactions = await self._staging.list_model_interactions(
                run_id=step_run_id
            )
            if interactions and isinstance(archive, StateStepArchive):
                prepared = await archive.prepare_interactions(
                    run,
                    tuple(interactions),
                    lambda digest: self._staging.staged_payload(
                        step_run_id,
                        digest,
                    ),
                    source_messages=snapshot.messages,
                )
                await archive.sync_projection(
                    run,
                    events=(),
                    snapshots=(),
                    interactions=prepared,
                )

    async def materialize_conversation(self, *, step_run_id: str) -> None:
        run = await self._staging.get_run(run_id=step_run_id)
        snapshot = await self._staging.latest_snapshot(run_id=step_run_id)
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if run is None or snapshot is None or archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await _materialize_snapshot(archive, run, snapshot)

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        destination = self._archives.get(target)
        if recovery is None or destination is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        run = await recovery.get_run(run_id=step_run_id)
        snapshot = await recovery.latest_snapshot(run_id=step_run_id)
        if run is None or snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if target is RuntimeDomain.EXECUTION and execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if target is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(step_run_id)
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.RECOVERY_MATERIALIZATION,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def operation() -> None:
                await _materialize_snapshot(
                    destination,
                    run,
                    snapshot,
                    execution_id=execution_id,
                )
                if isinstance(destination, StateStepArchive):
                    interactions = await recovery.list_model_interactions(
                        run_id=step_run_id
                    )
                    if interactions:
                        await destination.sync_projection(
                            run,
                            events=(),
                            snapshots=(),
                            interactions=tuple(
                                value
                                for value in interactions
                                if isinstance(value, ModelInteractionRecord)
                            ),
                            execution_id=execution_id,
                        )

            async def readback() -> CommitObservation[None]:
                try:
                    observed_run = await destination.get_run(run_id=run.run_id)
                    observed_snapshot = await destination.latest_snapshot(
                        run_id=run.run_id,
                        include_interrupted=True,
                    )
                    source_interactions = await recovery.list_model_interactions(
                        run_id=step_run_id
                    )
                    observed_interactions = await destination.list_model_interactions(
                        run_id=run.run_id
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if (
                    observed_run == run
                    and observed_snapshot == snapshot
                    and tuple(observed_interactions) == tuple(source_interactions)
                ):
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        await self._ensure_business()
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        ordered_run_ids = tuple(sorted(dict.fromkeys(run_ids)))
        if not ordered_run_ids:
            _logger.info(
                "execution terminal seal prepared: execution=%s runs=0",
                execution_id,
            )
            return ExecutionTerminalSealPlan(
                execution_id,
                binding_digest,
                (),
                (),
            )
        captured: list[tuple[_LocalExecutionTerminalSeal, ExecutionProjectionBatch]] = []
        installed: list[str] = []
        terminal_attempt_token = uuid4().hex
        try:
            for run_id in ordered_run_ids:
                while True:
                    archive_run: RunRecord | None = None
                    captured_seal: _LocalExecutionTerminalSeal | None = None
                    needs_archive_run = False
                    async with self._history_lock.hold(run_id):
                        existing = self._durability_flights.get(run_id)
                        if existing is not None:
                            completion = existing.completion
                        else:
                            seal = self._terminal_seals.get(run_id)
                            if seal is None:
                                seal = _LocalExecutionTerminalSeal(
                                    execution_id,
                                    terminal_attempt_token,
                                )
                                self._terminal_seals[run_id] = seal
                                self._install_durability_flight_locked(
                                    run_id,
                                    _RunDurabilityKind.TERMINAL,
                                    token=terminal_attempt_token,
                                )
                                installed.append(run_id)
                            elif seal.execution_id != execution_id or seal.token != terminal_attempt_token:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            captured_seal = seal
                            run = self._staging.get_run_local(run_id)
                            if run is None:
                                needs_archive_run = True
                            else:
                                projection = self._capture_projection_snapshot_locked(run_id)
                                captured.append((seal, projection))
                                break
                    if needs_archive_run:
                        archive_run = await archive.get_run(run_id=run_id)
                        if archive_run is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if captured_seal is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        captured.append(
                            (
                                captured_seal,
                                ExecutionProjectionBatch(
                                    archive_run,
                                    (),
                                    (),
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    (),
                                    0,
                                    0,
                                ),
                            )
                        )
                        break
                    await asyncio.shield(completion)
            prepared: list[PreparedExecutionProjection] = []
            durable_heads = await archive.execution_history_heads(
                tuple(projection.run.run_id for _seal, projection in captured)
            )
            for _seal, projection in captured:
                durable_head = durable_heads.get(projection.run.run_id)
                if durable_head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                batch = await archive.prepare_snapshots_after_seal(
                    projection.run,
                    projection.snapshots,
                )
                prepared.append(
                    PreparedExecutionProjection(
                        projection.run,
                        projection.events,
                        batch.snapshots,
                        projection.base_event_offset,
                        projection.base_snapshot_offset,
                        durable_head.event_count
                        + projection.target_event_offset
                        - projection.base_event_offset,
                        durable_head.snapshot_count
                        + projection.target_snapshot_offset
                        - projection.base_snapshot_offset,
                        batch.target_transcript_message_count,
                        "empty"
                        if not batch.snapshots
                        and durable_head.projection_digest == "empty"
                        else (
                            batch.snapshots[-1].projection.digest
                            if batch.snapshots
                            else durable_head.projection_digest
                        ),
                        tuple(
                            await archive.prepare_interactions(
                                projection.run,
                                projection.interactions,
                                lambda digest,
                                run_id=projection.run.run_id: self._staging.staged_payload(
                                    run_id,
                                    digest,
                                ),
                                source_messages=(
                                    projection.snapshots[-1].messages
                                    if projection.snapshots
                                    else None
                                ),
                            )
                        ),
                        durable_head.interaction_count
                        ,
                        durable_head.interaction_count
                        + projection.target_interaction_offset
                        - projection.base_interaction_offset,
                    )
                )
            plan = ExecutionTerminalSealPlan(
                execution_id,
                binding_digest,
                tuple(prepared),
                tuple(
                    (projection.run.run_id, seal.token)
                    for seal, projection in captured
                ),
                terminal_attempt_token,
            )
            _logger.info(
                "execution terminal seal prepared: execution=%s runs=%s",
                execution_id,
                len(plan.projections),
            )
            return plan
        except BaseException:
            for run_id in installed:
                await self._release_terminal_seal_if_owned(
                    run_id,
                    execution_id=execution_id,
                    token=terminal_attempt_token,
                )
            raise

    async def finalize_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        for projection in plan.projections:
            completion: asyncio.Future[None] | None = None
            async with self._history_lock.hold(projection.run.run_id):
                seal = self._terminal_seals.get(projection.run.run_id)
                if seal is None or seal.token != plan.token_for(
                    projection.run.run_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                flight = self._durability_flights.get(projection.run.run_id)
                if (
                    flight is None
                    or flight.kind is not _RunDurabilityKind.TERMINAL
                    or flight.token != seal.token
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                del self._durability_flights[projection.run.run_id]
                del self._terminal_seals[projection.run.run_id]
                completion = flight.completion
                offset = self._projection_offsets.setdefault(
                    projection.run.run_id,
                    _ProjectionOffset(),
                )
                offset.events = max(offset.events, projection.target_event_offset)
                offset.snapshots = max(offset.snapshots, projection.target_snapshot_offset)
                offset.transcript_messages = max(
                    offset.transcript_messages,
                    projection.target_transcript_message_count,
                )
                offset.interactions = max(
                    offset.interactions,
                    projection.target_interaction_offset,
                )
                self._projection_dirty.discard(projection.run.run_id)
            if completion is not None and not completion.done():
                completion.set_result(None)
        _logger.info(
            "execution terminal seal finalized: execution=%s runs=%s",
            plan.execution_id,
            len(plan.projections),
        )

    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        for run_id in dict.fromkeys(
            projection.run.run_id for projection in plan.projections
        ):
            await self._release_terminal_seal_if_owned(
                run_id,
                execution_id=plan.execution_id,
                token=plan.token_for(run_id),
            )

    async def _release_terminal_seal_if_owned(
        self,
        run_id: str,
        *,
        execution_id: str,
        token: str,
    ) -> None:
        """Release one run's terminal seal only when this attempt owns it."""
        completion: asyncio.Future[None] | None = None
        async with self._history_lock.hold(run_id):
            seal = self._terminal_seals.get(run_id)
            if seal is None:
                return
            if seal.execution_id != execution_id or seal.token != token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            flight = self._durability_flights.get(run_id)
            if flight is not None:
                if (
                    flight.kind is not _RunDurabilityKind.TERMINAL
                    or flight.token != token
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                del self._durability_flights[run_id]
                completion = flight.completion
            del self._terminal_seals[run_id]
        if completion is not None and not completion.done():
            completion.set_result(None)

    def _capture_projection_snapshot_locked(
        self,
        run_id: str,
    ) -> ExecutionProjectionBatch:
        offset = self._projection_offsets.setdefault(run_id, _ProjectionOffset())
        projection = self._staging.capture_projection_local(run_id, offset)
        if projection is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return projection

    def _ensure_run_mutable(self, run_id: str) -> None:
        if run_id in self._terminal_seals:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def capture_execution_projection(
        self,
        step_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _RunProjectionFlight] | None":
        """CAPTURE: snapshot staged state under the run lock with no durable I/O."""
        await self._ensure_business()
        while True:
            completion: asyncio.Future[None] | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(step_run_id)
                    offset = self._projection_offsets.setdefault(
                        step_run_id,
                        _ProjectionOffset(),
                    )
                    projection = self._staging.capture_projection_local(
                        step_run_id,
                        offset,
                    )
                    if projection is None:
                        return None
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.PROJECTION,
                    )
                    captured = CapturedExecutionProjection(
                        projection.run,
                        projection.events,
                        projection.snapshots,
                        projection.base_event_offset,
                        projection.base_snapshot_offset,
                        projection.target_event_offset,
                        projection.target_snapshot_offset,
                        projection.interactions,
                        projection.base_interaction_offset,
                        projection.target_interaction_offset,
                    )
                    _logger.debug(
                        "projection flight captured: run=%s token=%s "
                        "events=%s snapshots=%s",
                        step_run_id,
                        flight.token,
                        len(captured.events),
                        len(captured.snapshots),
                    )
                    return captured, flight
            await asyncio.shield(completion)

    async def wait_projection_flight(self, step_run_id: str) -> None:
        """Wait for an active flight without holding the run lock."""
        await self._ensure_business()
        while True:
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is None:
                    return
                completion = existing.completion
            await asyncio.shield(completion)

    async def abandon_execution_projection(
        self,
        flight: _RunProjectionFlight,
    ) -> None:
        """Remove a flight after a definitely-not-committed outcome."""
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "projection flight abandoned: run=%s token=%s",
            flight.run_id,
            flight.token,
        )

    async def finalize_execution_projection(
        self,
        flight: _RunProjectionFlight,
        captured: CapturedExecutionProjection,
        *,
        target_transcript_message_count: int | None = None,
    ) -> None:
        """FINALIZE: advance offsets and clear dirty state after durable success."""
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
            offset = self._projection_offsets.setdefault(
                flight.run_id,
                _ProjectionOffset(),
            )
            offset.events = max(offset.events, captured.target_event_offset)
            offset.snapshots = max(offset.snapshots, captured.target_snapshot_offset)
            offset.interactions = max(
                offset.interactions,
                captured.target_interaction_offset,
            )
            if target_transcript_message_count is not None:
                offset.transcript_messages = max(
                    offset.transcript_messages,
                    target_transcript_message_count,
                )
            self._projection_dirty.discard(flight.run_id)
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "projection flight finalized: run=%s token=%s events=%s snapshots=%s",
            flight.run_id,
            flight.token,
            captured.target_event_offset,
            captured.target_snapshot_offset,
        )

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _RunProjectionFlight,
        *,
        execution_id: str,
    ) -> None:
        """PREPARE + DURABLE COMMIT without the run lock, then FINALIZE."""
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            if not isinstance(archive, _StepArchiveBatch):
                await self.abandon_execution_projection(flight)
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

            async def operation() -> None:
                await _sync_projection(
                    archive,
                    captured.run,
                    captured.events,
                    captured.snapshots,
                    captured.interactions,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    stored_run = await archive.get_run(run_id=captured.run.run_id)
                    stored_events = await archive.list_events(
                        run_id=captured.run.run_id
                    )
                    stored_snapshot = await archive.latest_snapshot(
                        run_id=captured.run.run_id,
                        include_interrupted=True,
                    )
                    stored_interactions = await archive.list_model_interactions(
                        run_id=captured.run.run_id
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if stored_run != captured.run:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.events and tuple(stored_events[-len(captured.events) :]) != captured.events:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.snapshots and stored_snapshot != captured.snapshots[-1]:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.interactions and tuple(
                    stored_interactions[-len(captured.interactions) :]
                ) != tuple(captured.interactions):
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                return CommitObservation(DurableCommitState.COMMITTED)

            result = await run_durable_commit(
                operation,
                readback,
                background_tasks=self._background_tasks,
            )
            if result.state is DurableCommitState.COMMITTED:
                await self.finalize_execution_projection(flight, captured)
                if result.cancelled:
                    raise asyncio.CancelledError
                return
            if result.state is DurableCommitState.NOT_COMMITTED:
                await self.abandon_execution_projection(flight)
                if result.error is not None:
                    raise result.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                await self._fence_durability_flight(
                    flight,
                    AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        "projection commit left partial durable state",
                    ),
                )
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "projection commit left partial durable state",
                ) from result.error
            _logger.error(
                "projection commit unresolved; flight retained: run=%s token=%s",
                flight.run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        if not captured.events and not captured.snapshots and not captured.interactions:
            await self.finalize_execution_projection(flight, captured)
            return
        started = monotonic()
        try:
            prepared = await archive.prepare_snapshots(
                captured.run,
                captured.snapshots,
            )
            interactions = await archive.prepare_interactions(
                captured.run,
                captured.interactions,
                lambda digest: self._staging.staged_payload(
                    captured.run.run_id,
                    digest,
                ),
                source_messages=(
                    captured.snapshots[-1].messages
                    if captured.snapshots
                    else None
                ),
            )
        except BaseException:
            await self.abandon_execution_projection(flight)
            raise
        durable_head = await archive.execution_history_head_record(
            captured.run.run_id
        )
        expected_head = ExecutionRunSealHead(
            captured.run.run_id,
            captured.target_event_offset,
            captured.target_snapshot_offset,
            prepared.target_transcript_message_count,
            prepared.snapshots[-1].projection.digest
            if prepared.snapshots
            else "empty",
            durable_head.interaction_count + len(interactions),
        )

        async def operation() -> ExecutionRunSealHead:
            await archive.sync_prepared_projection(
                captured.run,
                events=captured.events,
                snapshots=prepared.snapshots,
                interactions=interactions,
                execution_id=execution_id,
            )
            head = await archive.execution_history_head_record(captured.run.run_id)
            if (
                head.event_count != expected_head.event_count
                or head.snapshot_count != expected_head.snapshot_count
                or head.transcript_message_count
                != expected_head.transcript_message_count
                or head.interaction_count != expected_head.interaction_count
                or (
                    prepared.snapshots
                    and head.projection_digest != expected_head.projection_digest
                )
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return ExecutionRunSealHead(
                captured.run.run_id,
                head.event_count,
                head.snapshot_count,
                head.transcript_message_count,
                head.projection_digest,
                head.interaction_count,
            )

        async def readback() -> CommitObservation[ExecutionRunSealHead]:
            try:
                head = await archive.execution_history_head_record(
                    captured.run.run_id
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            if head.event_count != expected_head.event_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.snapshot_count != expected_head.snapshot_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.transcript_message_count != expected_head.transcript_message_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.interaction_count != expected_head.interaction_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if prepared.snapshots and head.projection_digest != expected_head.projection_digest:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(
                DurableCommitState.COMMITTED,
                value=ExecutionRunSealHead(
                    captured.run.run_id,
                    head.event_count,
                    head.snapshot_count,
                    head.transcript_message_count,
                    head.projection_digest,
                    head.interaction_count,
                ),
            )

        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            if result.value is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self.finalize_execution_projection(
                flight,
                captured,
                target_transcript_message_count=result.value.transcript_message_count,
            )
            if result.cancelled:
                raise asyncio.CancelledError
        elif result.state is DurableCommitState.NOT_COMMITTED:
            await self.abandon_execution_projection(flight)
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            integrity = AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "projection commit left partial durable state",
            )
            await self._fence_durability_flight(flight, integrity)
            raise integrity from result.error
        else:
            _logger.error(
                "projection commit unresolved; flight retained: run=%s token=%s",
                flight.run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        _logger.debug(
            "step projection flushed: domain=%s backend=%s run=%s "
            "events=%s snapshots=%s duration_ms=%.3f",
            RuntimeDomain.EXECUTION.value,
            type(archive).__name__,
            flight.run_id,
            len(captured.events),
            len(captured.snapshots),
            (monotonic() - started) * 1000,
        )

    async def flush_execution_projection(
        self,
        step_run_id: str,
        *,
        execution_id: str,
    ) -> None:
        captured = await self.capture_execution_projection(step_run_id)
        if captured is None:
            return
        projection, flight = captured
        await self.commit_captured_execution_projection(
            projection,
            flight,
            execution_id=execution_id,
        )

    async def flush_dirty_execution_projections(self, *, execution_id: str) -> None:
        for run_id in tuple(self._projection_dirty):
            await self.flush_execution_projection(run_id, execution_id=execution_id)


    async def verify_terminal_attempts(
        self, *, candidate_step_run_ids: tuple[str, ...], required_step_run_id: str | None
    ) -> None:
        for run_id in dict.fromkeys(candidate_step_run_ids):
            if required_step_run_id != run_id:
                continue
            snapshot = await self._staging.latest_snapshot(run_id=run_id, include_interrupted=True)
            if snapshot is None:
                for archive in self._archives.values():
                    snapshot = await archive.latest_snapshot(
                        run_id=run_id,
                        include_interrupted=True,
                    )
                    if snapshot is not None:
                        break
            if snapshot is None or snapshot.state != "complete":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def release_staging_many(
        self,
        *,
        candidate_step_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None:
        for run_id in dict.fromkeys(candidate_step_run_ids):
            while True:
                completion: asyncio.Future[None] | None = None
                seal_owner: str | None = None
                seal_token: str | None = None
                async with self._history_lock.hold(run_id):
                    existing = self._durability_flights.get(run_id)
                    if existing is not None:
                        completion = existing.completion
                    else:
                        if run_id in self._projection_dirty:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        self._staging.release_run_local(run_id)
                        self._projection_offsets.pop(run_id, None)
                        self._projection_dirty.discard(run_id)
                        for archive in self._archives.values():
                            if isinstance(archive, StateStepArchive):
                                archive.release_runtime_cache(run_id)
                        seal = self._terminal_seals.get(run_id)
                        if seal is not None and seal.execution_id == execution_id:
                            seal_owner = seal.execution_id
                            seal_token = seal.token
                if completion is None:
                    break
                await asyncio.shield(completion)
            if seal_owner is not None and seal_token is not None:
                await self._release_terminal_seal_if_owned(
                    run_id,
                    execution_id=seal_owner,
                    token=seal_token,
                )
                _logger.warning(
                    "terminal seal discarded on staging release: run=%s "
                    "execution=%s",
                    run_id,
                    seal_owner,
                )

    async def release_archive(
        self,
        runtime_domain: RuntimeDomain,
        step_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        archive = self._archives.get(runtime_domain)
        if archive is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _RunDurabilityFlight | None = None
            async with self._history_lock.hold(step_run_id):
                existing = self._durability_flights.get(step_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if runtime_domain is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(step_run_id)
                    self._projection_offsets.pop(step_run_id, None)
                    self._projection_dirty.discard(step_run_id)
                    flight = self._install_durability_flight_locked(
                        step_run_id,
                        _RunDurabilityKind.RELEASE,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def operation() -> None:
                await archive.release_run(
                    step_run_id,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    observed = await archive.get_run(run_id=step_run_id)
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                return (
                    CommitObservation(DurableCommitState.COMMITTED)
                    if observed is None
                    else CommitObservation(DurableCommitState.NOT_COMMITTED)
                )

            await self._settle_durability_flight(flight, operation, readback)
            return

    def _install_durability_flight_locked(
        self,
        run_id: str,
        kind: _RunDurabilityKind,
        *,
        token: str | None = None,
    ) -> _RunDurabilityFlight:
        existing = self._durability_flights.get(run_id)
        if existing is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        flight = _RunDurabilityFlight(
            run_id,
            token or uuid4().hex,
            kind,
            asyncio.get_running_loop().create_future(),
        )
        self._durability_flights[run_id] = flight
        _logger.debug(
            "durability flight captured: run=%s token=%s kind=%s",
            run_id,
            flight.token,
            kind.value,
        )
        return flight

    async def _finalize_durability_flight(
        self,
        flight: _RunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "durability flight finalized: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )

    async def _abandon_durability_flight(
        self,
        flight: _RunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            if self._durability_flights.get(flight.run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "durability flight abandoned: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )

    async def _fence_durability_flight(
        self,
        flight: _RunDurabilityFlight,
        error: AIError,
    ) -> None:
        async with self._history_lock.hold(flight.run_id):
            current = self._durability_flights.get(flight.run_id)
            if current is not flight or current.token != flight.token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            completion = flight.completion
        if not completion.done():
            completion.set_exception(error)

            def consume(future: asyncio.Future[None]) -> None:
                future.exception()

            completion.add_done_callback(consume)
        _logger.error(
            "durability flight fenced: run=%s token=%s kind=%s code=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
            error.code.value,
        )

    async def _settle_durability_flight(
        self,
        flight: _RunDurabilityFlight,
        operation: Callable[[], Awaitable[None]],
        readback: Callable[[], Awaitable[CommitObservation[None]]],
    ) -> None:
        result = await run_durable_commit(
            operation,
            readback,
            background_tasks=self._background_tasks,
        )
        if result.state is DurableCommitState.COMMITTED:
            await self._finalize_durability_flight(flight)
            if result.cancelled:
                raise asyncio.CancelledError
            return
        if result.state is DurableCommitState.NOT_COMMITTED:
            await self._abandon_durability_flight(flight)
            if result.error is not None:
                raise result.error
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if result.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
            integrity = AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                "durability flight left partial durable state",
            )
            await self._fence_durability_flight(flight, integrity)
            raise integrity from result.error
        _logger.error(
            "durability flight unresolved: run=%s token=%s kind=%s",
            flight.run_id,
            flight.token,
            flight.kind.value,
        )
        unknown = AIError(
            ErrorCode.STORAGE_COMMIT_UNKNOWN,
            "durability flight commit outcome is unresolved",
        )
        await self._fence_durability_flight(flight, unknown)
        raise unknown from result.error

    async def preflight_close(self) -> None:
        pending_tasks = tuple(
            task for task in self._background_tasks if not task.done()
        )
        if self._durability_flights or pending_tasks or self._terminal_seals:
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "step_preflight_close",
                    "pending_flights": len(self._durability_flights),
                    "pending_tasks": len(pending_tasks),
                    "pending_terminal_seals": len(self._terminal_seals),
                },
            )
        self._preflight = True

    async def close(self) -> None:
        if not self._preflight:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._staging.close()
        for archive in self._archives.values():
            await archive.close()
        self._initialized = False

    async def _ensure_business(self) -> None:
        if not self._initialized:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


__all__ = [
    "ExecutionProjectionBatch",
    "ExecutionTerminalSealPlan",
    "InMemoryStepArchive",
    "LockOrderError",
    "PreparedExecutionProjection",
    "PreparedStepSnapshot",
    "PreparedStepSnapshotBatch",
    "RuntimeStepStore",
    "StagingStepStore",
    "StateStepArchive",
]
