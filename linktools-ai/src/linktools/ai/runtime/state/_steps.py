#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime AgentRunStore orchestration over durable archives."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import uuid4

from linktools.core import environ
from pydantic_ai.messages import ModelMessage

from ...errors import AIError, ErrorCode
from .._message import decode_model_messages, encode_model_messages
from .._transcript_staging import StagedTranscript
from .._model_interaction import (
    StagedContextInline,
    StagedContextSpan,
    StagedModelInteraction,
)
from ._contracts import (
    ExecutionRunSealHead,
    LoadedContextMessage,
    LoadedModelContext,
    ModelInteractionRecord,
    TranscriptMessageRef,
    TranscriptSpanRef,
)
from ._durability import CommitObservation, DurableCommitState, _await_owned_task, run_durable_commit
from ._plan import RuntimeDomain, RuntimeRetentionMode
from ._step_archive import (
    CapturedExecutionProjection,
    ExecutionProjectionBatch,
    ExecutionTerminalSealPlan,
    InMemoryStepArchive,
    LockOrderError,
    PreparedExecutionProjection,
    PreparedAgentRunCheckpoint,
    PreparedAgentRunCheckpointBatch,
    StagingAgentRunStore,
    StateStepArchive,
    _LocalExecutionTerminalSeal,
    _ProjectionOffset,
    _AgentRunDurabilityFlight,
    _AgentRunDurabilityKind,
    _AgentRunHistoryLock,
    _AgentRunProjectionFlight,
    _StepArchiveBatch,
    _conversation_relocated_checkpoint_matches,
    _materialize_checkpoint,
    _sync_projection,
)
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent, AgentRunStore
from ._store import StateGroupTransaction, StateTransaction

if TYPE_CHECKING:
    from ..service_api import ModelInteractionSubscription

_logger = environ.get_logger("ai.runtime.state.run_store")
_OBSERVATION_FLUSH_SECONDS = 1.0
_OBSERVATION_FLUSH_BOUNDARIES = 16
_BoundaryResult = TypeVar("_BoundaryResult")


class _ModelInteractionSubscription:
    def __init__(self, remove: Callable[["_ModelInteractionSubscription"], None]) -> None:
        self._remove = remove
        self._generation = 0
        self._changed = asyncio.Event()
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    def publish(self) -> None:
        if not self._closed:
            self._generation += 1
            self._changed.set()

    async def wait(self, after_generation: int) -> int:
        if (isinstance(after_generation, bool) or not isinstance(after_generation, int)
                or after_generation < 0 or after_generation > self._generation):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        while self._generation <= after_generation and not self._closed:
            self._changed.clear()
            await self._changed.wait()
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=False)
        return self._generation

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._remove(self)
            self._changed.set()


class RuntimeAgentRunStore(AgentRunStore):
    """Route staging facts to their owning durable StateStore archive."""

    def __init__(
        self,
        staging: StagingAgentRunStore,
        *,
        conversation_archive: AgentRunStore,
        execution_archive: AgentRunStore | None,
        recovery_archive: AgentRunStore | None,
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
        self._execution_producers: dict[str, tuple[str, int]] = {}
        self._projection_dirty: set[str] = set()
        self._observation_task: asyncio.Task[None] | None = None
        self._observation_wake = asyncio.Event()
        self._observation_started_at: float | None = None
        self._model_interaction_subscriptions: dict[str, set[_ModelInteractionSubscription]] = {}
        self._durability_flights: dict[str, _AgentRunDurabilityFlight] = {}
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._terminal_seals: dict[str, _LocalExecutionTerminalSeal] = {}
        self._history_lock = _AgentRunHistoryLock()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.bind_history_lock(self._history_lock)

    def bind_execution_producer(
        self, agent_run_id: str, *, execution_id: str, producer_generation: int,
    ) -> None:
        if not agent_run_id or not execution_id or isinstance(producer_generation, bool) or producer_generation < 0:
            raise ValueError("execution producer identity is invalid")
        value = (execution_id, producer_generation)
        previous = self._execution_producers.get(agent_run_id)
        if previous is not None and (previous[0] != execution_id or previous[1] > producer_generation):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        self._execution_producers[agent_run_id] = value

    def execution_producer_generation(self, agent_run_id: str, *, execution_id: str) -> int | None:
        producer = self._execution_producers.get(agent_run_id)
        if producer is None:
            return None
        if producer[0] != execution_id:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return producer[1]

    async def initialize(self) -> None:
        await self._staging.initialize()
        for archive in self._archives.values():
            await archive.initialize()
        self._projection_offsets.clear()
        self._projection_dirty.clear()
        self._observation_task = None
        self._observation_wake.clear()
        self._observation_started_at = None
        self._durability_flights.clear()
        self._terminal_seals.clear()
        self._preflight = False
        self._initialized = True

    async def validate_integrity(self) -> None:
        await self._ensure_business()
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                await archive.validate_integrity()

    def register_context_baseline(
        self,
        agent_run_id: str,
        context: LoadedModelContext,
    ) -> None:
        for archive in self._archives.values():
            if isinstance(archive, StateStepArchive):
                archive.register_context_baseline(agent_run_id, context)

    async def register_agent_run(
        self,
        record: AgentRunRecord,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        restored: AgentRunCheckpoint | None = None
        observed_transcript: StagedTranscript | None = None
        events: Sequence[StepEvent] = ()
        offset: _ProjectionOffset | None = None
        if recovery is not None:
            durable = await recovery.get_agent_run(agent_run_id=record.agent_run_id)
            if durable is not None:
                if _agent_run_identity(durable) != _agent_run_identity(record):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                record = durable
                restored = await recovery.latest_checkpoint(
                    agent_run_id=record.agent_run_id, include_interrupted=True,
                )
        execution = self._archives.get(RuntimeDomain.EXECUTION)
        if execution is not None and execution_id is not None:
            public_run = await execution.get_agent_run(agent_run_id=record.agent_run_id)
            if restored is not None and self._staging.get_agent_run_local(record.agent_run_id) is None:
                published = tuple(
                    [message async for message in execution.iter_messages(agent_run_id=record.agent_run_id)]
                ) if public_run is not None else ()
                common = min(len(published), len(restored.messages))
                if encode_model_messages(published[:common]) != encode_model_messages(restored.messages[:common]):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if public_run is None or len(published) <= len(restored.messages):
                    # Restore known completions before interrupting observations
                    # whose request really was left unfinished by the old owner.
                    await self.materialize_from_recovery(
                        target=RuntimeDomain.EXECUTION, agent_run_id=record.agent_run_id,
                        execution_id=execution_id,
                    )
                    public_run = await execution.get_agent_run(agent_run_id=record.agent_run_id)
            if public_run is not None:
                if _agent_run_identity(public_run) != _agent_run_identity(record):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                producer = self._execution_producers.get(record.agent_run_id)
                if (isinstance(execution, StateStepArchive) and producer is not None
                        and self._staging.get_agent_run_local(record.agent_run_id) is None):
                    await execution.interrupt_model_interactions(
                        agent_run_id=record.agent_run_id, execution_id=producer[0],
                        producer_generation=producer[1],
                    )
                events = await execution.list_events(agent_run_id=record.agent_run_id)
                messages = tuple([message async for message in execution.iter_messages(agent_run_id=record.agent_run_id)])
                pending = None
                pending_keys = ()
                if isinstance(execution, StateStepArchive):
                    head = await execution.transcript_repository.get_head(record.agent_run_id)
                    if head is not None:
                        pending, pending_keys = await execution.transcript_repository.load_pending(head)
                observed_transcript = StagedTranscript(messages, pending, pending_keys)
                checkpoint_published = restored is not None and (
                    await execution.verify_checkpoint_projection(
                        agent_run_id=record.agent_run_id, checkpoint=restored,
                    ) if isinstance(execution, StateStepArchive)
                    else await execution.latest_checkpoint(
                        agent_run_id=record.agent_run_id, include_interrupted=True,
                    ) == restored
                )
                offset = _ProjectionOffset(
                    events=len(events), checkpoints=int(checkpoint_published),
                    transcript_messages=len(messages),
                    interactions=await execution.model_interaction_count(agent_run_id=record.agent_run_id),
                )
        async with self._history_lock.hold(record.agent_run_id):
            self._ensure_run_mutable(record.agent_run_id)
            new_registration = self._staging.get_agent_run_local(record.agent_run_id) is None
            self._staging.register_agent_run_local(record)
            if new_registration:
                if restored is not None:
                    self._staging.save_checkpoint_local(replace(restored, transcript_message_count_before=0))
                if observed_transcript is not None:
                    self._staging.stage_transcript(record.agent_run_id, observed_transcript)
                for event in events:
                    self._staging.append_event_local(event)
                if offset is not None:
                    self._projection_offsets[record.agent_run_id] = offset
                if offset is None or restored is not None and offset.checkpoints == 0:
                    self._projection_dirty.add(record.agent_run_id)
                    producer = self._execution_producers.get(record.agent_run_id)
                    if offset is not None and producer is not None:
                        offset.observation_execution_id, offset.observation_producer_generation = producer
        self._publish_model_interaction_change(record.agent_run_id)

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        await self._ensure_business()
        return await self._staging.get_agent_run(agent_run_id=agent_run_id)

    async def list_agent_runs(
        self, *, parent_agent_run_id: str | None = None, agent_conversation_id: str | None = None
    ) -> list[AgentRunRecord]:
        await self._ensure_business()
        return await self._staging.list_agent_runs(parent_agent_run_id=parent_agent_run_id, agent_conversation_id=agent_conversation_id)

    async def append_event(
        self,
        event: StepEvent,
        *,
        execution_id: str | None = None,
    ) -> None:
        await self._ensure_business()
        del execution_id
        async with self._history_lock.hold(event.agent_run_id):
            self._ensure_run_mutable(event.agent_run_id)
            self._staging.append_event_local(event)
            self._projection_dirty.add(event.agent_run_id)

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        await self._ensure_business()
        return await self._staging.list_events(agent_run_id=agent_run_id)

    async def save_checkpoint(
        self, checkpoint: AgentRunCheckpoint, *, execution_id: str | None = None,
        producer_generation: int | None = None,
    ) -> None:
        await self._settle_observation_boundary(self._save_checkpoint(
            checkpoint, execution_id=execution_id, producer_generation=producer_generation,
        ))

    async def _save_checkpoint(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
        producer_generation: int | None = None,
    ) -> None:
        await self._ensure_business()
        execution = (
            self._archives.get(RuntimeDomain.EXECUTION)
            if execution_id is not None and producer_generation is not None else None
        )
        if execution is not None:
            recovery = self._archives.get(RuntimeDomain.RECOVERY)
            if (not isinstance(execution, StateStepArchive)
                    or not isinstance(recovery, StateStepArchive)
                    or execution.state_store.storage_group is not recovery.state_store.storage_group):
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        while True:
            completion: asyncio.Future[None] | None = None
            recovery: AgentRunStore | None = None
            recovery_run: AgentRunRecord | None = None
            recovery_interactions_snapshot: tuple[StagedModelInteraction, ...] = ()
            flight: _AgentRunDurabilityFlight | None = None
            async with self._history_lock.hold(checkpoint.agent_run_id):
                existing = self._durability_flights.get(checkpoint.agent_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(checkpoint.agent_run_id)
                    if execution is not None:
                        offset = self._projection_offsets.setdefault(checkpoint.agent_run_id, _ProjectionOffset())
                        if (offset.observation_execution_id is not None
                                and (offset.observation_execution_id != execution_id
                                     or offset.observation_producer_generation != producer_generation)):
                            raise AIError(ErrorCode.STORAGE_CONFLICT)
                        offset.observation_execution_id = execution_id
                        offset.observation_producer_generation = producer_generation
                    self._staging.save_checkpoint_local(checkpoint)
                    self._projection_dirty.add(checkpoint.agent_run_id)
                    recovery = self._archives.get(RuntimeDomain.RECOVERY)
                    recovery_run = self._staging.get_agent_run_local(checkpoint.agent_run_id)
                    recovery_interactions_snapshot = self._staging.capture_interactions_local(checkpoint.agent_run_id)
                    if recovery is not None:
                        flight = self._install_durability_flight_locked(
                            checkpoint.agent_run_id,
                            _AgentRunDurabilityKind.CHECKPOINT,
                        )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if recovery is None:
                return
            if recovery_run is None or flight is None:
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)

            prepared_interactions: tuple[ModelInteractionRecord, ...] = ()
            prepared_checkpoint: PreparedAgentRunCheckpoint | None = None
            prepared_sequence = 0
            cancellation_requested = False

            async def operation(
                target_recovery: AgentRunStore = recovery,
                target_agent_run: AgentRunRecord = recovery_run,
                target_checkpoint: AgentRunCheckpoint = checkpoint,
            ) -> None:
                nonlocal prepared_interactions, prepared_checkpoint, prepared_sequence, cancellation_requested
                if isinstance(target_recovery, StateStepArchive):
                    prepared_checkpoint, prepared_interactions, prepared_sequence = await self._prepare_recovery_checkpoint(
                        target_recovery, target_agent_run, target_checkpoint, recovery_interactions_snapshot,
                    )
                    async def commit(transaction: StateTransaction) -> None:
                        await target_recovery.materialize_checkpoint_in_transaction(
                            transaction, target_agent_run, prepared_checkpoint,
                        )
                        if prepared_interactions:
                            await target_recovery.sync_projection_in_transaction(
                                transaction, target_agent_run, events=(), checkpoints=(),
                                interactions=prepared_interactions,
                            )
                    if isinstance(execution, StateStepArchive):
                        async def guarded_commit(group: StateGroupTransaction) -> None:
                            nonlocal cancellation_requested
                            try:
                                await execution.guard_execution_producer(
                                    group.transaction(execution.state_store),
                                    execution_id=execution_id,
                                    producer_generation=producer_generation,
                                    allow_cancelling=checkpoint.state == "interrupted",
                                )
                            except AIError as error:
                                if error.code is ErrorCode.EXECUTION_CANCELLED:
                                    cancellation_requested = True
                                raise
                            await commit(group.transaction(target_recovery.state_store))

                        await execution.state_store.storage_group.mutate(
                            (execution.state_store, target_recovery.state_store), guarded_commit,
                        )
                    else:
                        await target_recovery.state_store.mutate(commit)
                    return
                await _materialize_checkpoint(
                    target_recovery,
                    target_agent_run,
                    target_checkpoint,
                )

            async def readback(
                target_recovery: AgentRunStore = recovery,
                target_agent_run: AgentRunRecord = recovery_run,
                target_checkpoint: AgentRunCheckpoint = checkpoint,
            ) -> CommitObservation[None]:
                try:
                    observed_run = await target_recovery.get_agent_run(
                        agent_run_id=target_checkpoint.agent_run_id
                    )
                    if isinstance(target_recovery, StateStepArchive):
                        checkpoint_visible = prepared_checkpoint is not None and (
                            await target_recovery.verify_checkpoint_range(
                                agent_run_id=target_checkpoint.agent_run_id,
                                after_sequence=prepared_sequence - 1,
                                checkpoints=(prepared_checkpoint,),
                            )
                        )
                        if checkpoint_visible and prepared_interactions:
                            observed_interactions = await target_recovery.list_model_interactions(
                                agent_run_id=target_agent_run.agent_run_id,
                                after_model_request_seq=prepared_interactions[0].model_request_seq - 1,
                                limit=len(prepared_interactions),
                            )
                            checkpoint_visible = tuple(observed_interactions) == prepared_interactions
                    else:
                        observed_checkpoint = await target_recovery.latest_checkpoint(
                            agent_run_id=target_checkpoint.agent_run_id,
                            include_interrupted=True,
                        )
                        checkpoint_visible = _relocated_checkpoint_matches(
                            RuntimeDomain.RECOVERY,
                            target_checkpoint,
                            observed_checkpoint,
                        )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if observed_run == target_agent_run and checkpoint_visible:
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            if cancellation_requested:
                raise AIError(ErrorCode.EXECUTION_CANCELLED)
            return

    async def _prepare_recovery_checkpoint(
        self, archive: StateStepArchive, run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        captured_interactions: Sequence[StagedModelInteraction] | None = None,
    ) -> tuple[PreparedAgentRunCheckpoint, tuple[ModelInteractionRecord, ...], int]:
        checkpoint = await archive.relocate_run_checkpoint(run, checkpoint)
        batch = await archive.prepare_checkpoints(run, (checkpoint,))
        prepared = batch.checkpoints[0]
        sequence = await archive.checkpoint_count(agent_run_id=run.agent_run_id)
        if not sequence or not await archive.verify_checkpoint_range(
            agent_run_id=run.agent_run_id, after_sequence=sequence - 1,
            checkpoints=(prepared,),
        ):
            sequence += 1
        high_water = await archive.model_interaction_count(agent_run_id=run.agent_run_id)
        if captured_interactions is None:
            staged = await self._terminal_staged_interactions(
                agent_run_id=run.agent_run_id, after_model_request_seq=high_water,
            )
        else:
            terminal: list[StagedModelInteraction] = []
            for interaction in captured_interactions:
                if interaction.model_request_seq <= high_water:
                    continue
                if interaction.status == "RUNNING":
                    break
                terminal.append(interaction)
            staged = tuple(terminal)
        missing: tuple[ModelInteractionRecord, ...] = ()
        if staged and staged[0].model_request_seq > high_water + 1:
            source = self._archives.get(RuntimeDomain.EXECUTION)
            if not isinstance(source, StateStepArchive):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            archived = await source.list_model_interactions(
                agent_run_id=run.agent_run_id, after_model_request_seq=high_water,
                limit=staged[0].model_request_seq - high_water - 1,
            )
            if any(not isinstance(value, ModelInteractionRecord) or value.status == "RUNNING" for value in archived):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            missing = await archive.prepare_relocated_interactions(
                archived, await source.resolve_model_interactions(archived),
            )
        interactions = missing + await archive.prepare_interactions(
            run, staged,
            lambda digest: self._staging.staged_payload(run.agent_run_id, digest),
            local_message_count=len(checkpoint.messages),
        )
        return prepared, interactions, sequence

    async def latest_checkpoint(self, *, agent_run_id: str, include_interrupted: bool = False) -> AgentRunCheckpoint | None:
        await self._ensure_business()
        return await self._staging.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=include_interrupted)

    def stage_transcript(self, agent_run_id: str, transcript: StagedTranscript) -> None:
        self._staging.stage_transcript(agent_run_id, transcript)
        self._projection_dirty.add(agent_run_id)

    def staged_transcript(self, agent_run_id: str) -> StagedTranscript | None:
        return self._staging.staged_transcript(agent_run_id)

    def intern_payload(self, agent_run_id: str, payload: bytes) -> tuple[str, int]:
        return self._staging.intern_payload(agent_run_id, payload)

    def staged_payload(self, agent_run_id: str, digest: str) -> bytes:
        return self._staging.staged_payload(agent_run_id, digest)

    def stage_model_interaction(self, interaction: object) -> None:
        self._staging.stage_model_interaction(interaction)
        agent_run_id = getattr(interaction, "agent_run_id", None)
        if not isinstance(agent_run_id, str) or not agent_run_id:
            raise TypeError("staged model interaction has no AgentRun identity")
        self._projection_dirty.add(agent_run_id)
        if RuntimeDomain.EXECUTION not in self._archives:
            self._publish_model_interaction_change(agent_run_id)

    def prepare_model_interaction(self, interaction: object) -> None:
        self._staging.prepare_model_interaction(interaction)
        if not isinstance(interaction, StagedModelInteraction):
            raise TypeError("staged model interaction is invalid")
        self._projection_dirty.add(interaction.agent_run_id)
        if RuntimeDomain.EXECUTION not in self._archives:
            self._publish_model_interaction_change(interaction.agent_run_id)

    @property
    def model_interaction_history_available(self) -> bool:
        """Whether execution history has an archive separate from local staging."""
        return RuntimeDomain.EXECUTION in self._archives

    def subscribe_model_interactions(
        self, agent_conversation_id: str,
    ) -> "ModelInteractionSubscription":
        if not self._initialized or self._preflight:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=False)
        if not isinstance(agent_conversation_id, str) or not agent_conversation_id:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        def remove(subscription: _ModelInteractionSubscription) -> None:
            subscribers = self._model_interaction_subscriptions.get(agent_conversation_id)
            if subscribers is not None:
                subscribers.discard(subscription)
                if not subscribers:
                    self._model_interaction_subscriptions.pop(agent_conversation_id, None)

        subscription = _ModelInteractionSubscription(remove)
        self._model_interaction_subscriptions.setdefault(agent_conversation_id, set()).add(subscription)
        return subscription

    def _publish_model_interaction_change(self, agent_run_id: str) -> None:
        run = self._staging.get_agent_run_local(agent_run_id)
        if run is not None and run.agent_conversation_id is not None:
            for subscription in tuple(self._model_interaction_subscriptions.get(run.agent_conversation_id, ())):
                subscription.publish()

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        await self._ensure_business()
        return await self._staging.list_model_interactions(
            agent_run_id=agent_run_id,
            after_model_request_seq=after_model_request_seq,
            limit=limit,
        )

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        await self._ensure_business()
        high_water = await self._staging.model_interaction_count(agent_run_id=agent_run_id)
        for domain in (RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
            archive = self._archives.get(domain)
            if archive is not None:
                high_water = max(high_water, await archive.model_interaction_count(agent_run_id=agent_run_id))
        return high_water

    async def _terminal_staged_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
    ) -> tuple[StagedModelInteraction, ...]:
        staged = await self._staging.list_model_interactions(
            agent_run_id=agent_run_id,
            after_model_request_seq=after_model_request_seq,
        )
        terminal: list[StagedModelInteraction] = []
        previous_sequence: int | None = None
        for value in staged:
            if not isinstance(value, StagedModelInteraction):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if previous_sequence is not None and value.model_request_seq != previous_sequence + 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_sequence = value.model_request_seq
            if value.status == "RUNNING":
                break
            terminal.append(value)
        return tuple(terminal)

    async def resolve_model_interaction(self, interaction: object) -> object:
        values = await self.resolve_model_interactions((interaction,))
        if len(values) != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values[0]

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        await self._ensure_business()
        values = tuple(interactions)
        if not values:
            return []
        if all(isinstance(value, StagedModelInteraction) for value in values):
            staged = tuple(
                value for value in values if isinstance(value, StagedModelInteraction)
            )
            agent_run_ids = {value.agent_run_id for value in staged}
            if len(agent_run_ids) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            agent_run_id = staged[0].agent_run_id
            transcript = self._staging.staged_transcript(agent_run_id)
            local_messages = (() if transcript is None else transcript.messages)

            async def resolve_projection(projection):
                messages: list[ModelMessage] = []
                for item in projection.items:
                    if isinstance(item, StagedContextSpan):
                        if item.end > len(local_messages):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        messages.extend(local_messages[item.start : item.end])
                    elif isinstance(item, StagedContextInline):
                        decoded = decode_model_messages(
                            self._staging.staged_payload(
                                agent_run_id,
                                item.payload_digest,
                            )
                        )
                        messages.extend(decoded)
                    elif isinstance(item, TranscriptSpanRef):
                        refs = tuple(
                            TranscriptMessageRef(
                                item.source_domain,
                                item.owner_id,
                                index,
                            )
                            for index in range(item.start, item.end)
                        )
                        resolved = await self.resolve_transcript_message_refs(refs)
                        messages.extend(value.message for value in resolved)
                    else:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return tuple(messages)

            try:
                resolved_values: list[object] = []
                for value in staged:
                    request = None if value.request_context is None else await resolve_projection(value.request_context)
                    response = (
                        None
                        if value.response_context is None
                        else await resolve_projection(value.response_context)
                    )
                    resolved_values.append(
                        (
                            request,
                            response,
                            None if value.request_envelope_digest is None else self._staging.staged_payload(
                                agent_run_id, value.request_envelope_digest,
                            ),
                        )
                    )
                return resolved_values
            except AIError:
                if self._staging.get_agent_run_local(agent_run_id) is not None:
                    raise
                archive = self.read_store(RuntimeDomain.EXECUTION)
                archived = await archive.list_model_interactions(
                    agent_run_id=agent_run_id,
                    after_model_request_seq=staged[0].model_request_seq - 1,
                    limit=len(staged),
                )
                if (
                    len(archived) != len(staged)
                    or any(
                        not isinstance(record, ModelInteractionRecord)
                        or record.agent_run_id != agent_run_id
                        or record.model_request_seq != staged_value.model_request_seq
                        for record, staged_value in zip(archived, staged, strict=True)
                    )
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                resolved = await archive.resolve_model_interactions(archived)
                if len(resolved) != len(staged):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result: list[object] = []
                for staged_value, value in zip(staged, resolved, strict=True):
                    if not isinstance(value, tuple) or len(value) != 3:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    request, response, envelope = value
                    result.append(
                        (
                            request,
                            response
                            if staged_value.response_context is not None
                            else None,
                            envelope,
                        )
                    )
                return result
        return await self._staging.resolve_model_interactions(values)

    def read_store(self, runtime_domain: RuntimeDomain) -> AgentRunStore:
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
        if isinstance(archive, (StateStepArchive, StagingAgentRunStore)):
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
        agent_run_id: str,
        tenant_id: str,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_messages(
            history_id=history_id,
            agent_run_id=agent_run_id,
            tenant_id=tenant_id,
        )

    async def _iter_conversation_messages(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
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
            async for message in archive.iter_messages(agent_run_id=agent_run_id):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def conversation_message_count(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
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
            return await archive.transcript_message_count(agent_run_id)
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    def iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        return self._iter_conversation_message_range(
            history_id=history_id,
            agent_run_id=agent_run_id,
            tenant_id=tenant_id,
            start=start,
            end=end,
        )

    async def _iter_conversation_message_range(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
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
                agent_run_id=agent_run_id,
                start=start,
                end=end,
            ):
                yield message
            return
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def load_committed_conversation_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        message_count: int | None,
        tenant_id: str,
    ) -> LoadedModelContext:
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(archive, StateStepArchive):
            if history_id is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if message_count is None:
                return await archive.load_loaded_model_context(
                    owner_id=history_id,
                )
            return await archive.load_committed_session_model_context(
                history_id,
                agent_run_id=agent_run_id,
                message_count=message_count,
                tenant_id=tenant_id,
            )
        if isinstance(archive, InMemoryStepArchive):
            values = await archive.load_model_context(agent_run_id=agent_run_id)
            return LoadedModelContext(
                tuple(
                    LoadedContextMessage(value, None)
                    for value in values
                    if isinstance(value, ModelMessage)
                )
            )
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)

    async def load_conversation_model_context(
        self,
        *,
        history_id: str | None,
        agent_run_id: str,
        tenant_id: str,
        message_count: int | None = None,
    ) -> tuple[object, ...]:
        context = await self.load_committed_conversation_context(
            history_id=history_id,
            agent_run_id=agent_run_id,
            message_count=message_count,
            tenant_id=tenant_id,
        )
        return context.model_messages()

    async def materialize_recovery_checkpoint(self, *, agent_run_id: str, require_complete: bool) -> None:
        checkpoint = await self._staging.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        run = await self._staging.get_agent_run(agent_run_id=agent_run_id)
        archive = self._archives.get(RuntimeDomain.RECOVERY)
        if checkpoint is None or run is None:
            if require_complete:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        if require_complete and checkpoint.state != "complete":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if archive is not None:
            if isinstance(archive, StateStepArchive):
                checkpoint = await archive.relocate_run_checkpoint(run, checkpoint)
            await _materialize_checkpoint(archive, run, checkpoint)
            interactions = await self._terminal_staged_interactions(
                agent_run_id=agent_run_id
            )
            if interactions and isinstance(archive, StateStepArchive):
                head = await archive.transcript_repository.get_head(run.agent_run_id)
                if head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                local_base, local_count = _interaction_local_range(
                    (checkpoint,),
                    head.message_count,
                )
                prepared = await archive.prepare_interactions(
                    run,
                    tuple(interactions),
                    lambda digest: self._staging.staged_payload(
                        agent_run_id,
                        digest,
                    ),
                    local_message_base=local_base,
                    local_message_count=local_count,
                )
                await archive.sync_projection(
                    run,
                    events=(),
                    checkpoints=(),
                    interactions=prepared,
                )

    async def materialize_conversation(self, *, agent_run_id: str) -> None:
        run = await self._staging.get_agent_run(agent_run_id=agent_run_id)
        checkpoint = await self._staging.latest_checkpoint(agent_run_id=agent_run_id)
        archive = self._archives.get(RuntimeDomain.CONVERSATION)
        if run is None or checkpoint is None or archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if isinstance(archive, StateStepArchive):
            checkpoint = await archive.relocate_conversation_checkpoint(
                run,
                checkpoint,
            )
        await _materialize_checkpoint(archive, run, checkpoint)

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        agent_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        recovery = self._archives.get(RuntimeDomain.RECOVERY)
        destination = self._archives.get(target)
        producer = self._execution_producers.get(agent_run_id) if target is RuntimeDomain.EXECUTION else None
        if recovery is None or destination is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        run = await recovery.get_agent_run(agent_run_id=agent_run_id)
        checkpoint = await recovery.latest_checkpoint(
            agent_run_id=agent_run_id,
            include_interrupted=target is RuntimeDomain.EXECUTION,
        )
        if run is None or checkpoint is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if target is RuntimeDomain.EXECUTION and execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _AgentRunDurabilityFlight | None = None
            async with self._history_lock.hold(agent_run_id):
                existing = self._durability_flights.get(agent_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if target is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(agent_run_id)
                    flight = self._install_durability_flight_locked(
                        agent_run_id,
                        _AgentRunDurabilityKind.RECOVERY_MATERIALIZATION,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            prepared_projection: PreparedAgentRunCheckpointBatch | None = None
            restored_observation = None
            try:
                source_values = await recovery.list_model_interactions(
                    agent_run_id=agent_run_id
                )
                source_interactions = tuple(
                    value
                    for value in source_values
                    if isinstance(value, ModelInteractionRecord)
                )
                if len(source_interactions) != len(source_values):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                source_resolved = tuple(
                    await recovery.resolve_model_interactions(source_interactions)
                )
                if not isinstance(destination, (StateStepArchive, InMemoryStepArchive)):
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
                if isinstance(destination, StateStepArchive):
                    if target is RuntimeDomain.CONVERSATION:
                        target_checkpoint = await destination.relocate_conversation_checkpoint(
                            run, checkpoint
                        )
                    else:
                        target_checkpoint = await destination.relocate_run_checkpoint(
                            run, checkpoint
                        )
                else:
                    local_message_count = await destination.transcript_message_count(
                        agent_run_id
                    )
                    if local_message_count > len(checkpoint.messages):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    target_checkpoint = replace(
                        checkpoint,
                        transcript_message_count_before=local_message_count,
                    )
                relocated = await destination.prepare_relocated_interactions(
                    source_interactions,
                    source_resolved,
                )
                existing_values = await destination.list_model_interactions(
                    agent_run_id=agent_run_id
                )
                existing_interactions = tuple(
                    value
                    for value in existing_values
                    if isinstance(value, ModelInteractionRecord)
                )
                if len(existing_interactions) != len(existing_values):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                existing_by_sequence = {
                    value.model_request_seq: value for value in existing_interactions
                }
                source_by_sequence = {
                    value.model_request_seq: (value, resolved)
                    for value, resolved in zip(
                        source_interactions,
                        source_resolved,
                        strict=True,
                    )
                }
                if len(existing_by_sequence) != len(existing_interactions):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                missing_relocated: list[ModelInteractionRecord] = []
                for value in relocated:
                    existing_value = existing_by_sequence.get(
                        value.model_request_seq
                    )
                    if existing_value is None:
                        missing_relocated.append(value)
                        continue
                    source_value, source_projection = source_by_sequence[
                        value.model_request_seq
                    ]
                    existing_projection = (
                        await destination.resolve_model_interactions((existing_value,))
                    )[0]
                    if existing_value.status == "RUNNING" and source_value.status != "RUNNING":
                        if existing_value.request_context is not None:
                            if (existing_projection[0], existing_projection[2]) != (source_projection[0], source_projection[2]):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            value = replace(
                                value, request_context=existing_value.request_context,
                                request_envelope=existing_value.request_envelope,
                            )
                        existing_value.validate_successor(value)
                        missing_relocated.append(value)
                    elif (
                        _interaction_semantic_header(existing_value) != _interaction_semantic_header(source_value)
                        or existing_projection != source_projection
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                relocated = tuple(missing_relocated)
                checkpoint_published = _relocated_checkpoint_matches(
                    target, checkpoint, await destination.latest_checkpoint(
                        agent_run_id=agent_run_id, include_interrupted=True,
                    ),
                )
                if checkpoint_published and not relocated:
                    await self._finalize_durability_flight(flight)
                    return
                if target is RuntimeDomain.EXECUTION and isinstance(destination, StateStepArchive):
                    public_head = await destination.transcript_repository.get_head(agent_run_id)
                    published_count = 0 if public_head is None else public_head.message_count
                    if published_count < len(target_checkpoint.messages):
                        restored_observation = await destination.transcript_repository.prepare_observation(
                            agent_run_id, target_checkpoint.messages[published_count:],
                            first_message_index=published_count, pending=None, pending_keys=(),
                        )
                        prepared_projection = await destination.prepare_checkpoints(
                            run, (target_checkpoint,),
                            observed_message_count=restored_observation.target_message_count,
                        )
            except BaseException:
                await self._abandon_durability_flight(flight)
                raise

            async def operation() -> None:
                async def publish() -> None:
                    if prepared_projection is not None and isinstance(destination, StateStepArchive):
                        await destination.sync_prepared_projection(
                            run, events=(), checkpoints=prepared_projection.checkpoints,
                            interactions=relocated, observation=restored_observation, execution_id=execution_id,
                        )
                    else:
                        await destination.sync_projection(
                            run, events=(), checkpoints=() if checkpoint_published else (target_checkpoint,),
                            interactions=relocated, execution_id=execution_id,
                        )

                if target is RuntimeDomain.EXECUTION and isinstance(destination, StateStepArchive) and producer is not None:
                    if producer[0] != execution_id:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
                    async def guarded_publish(group: StateGroupTransaction) -> None:
                        await destination.guard_execution_producer(
                            group.transaction(destination.state_store),
                            execution_id=producer[0], producer_generation=producer[1],
                        )
                        await publish()
                    await destination.state_store.storage_group.mutate((destination.state_store,), guarded_publish)
                else:
                    await publish()

            async def readback() -> CommitObservation[None]:
                try:
                    observed_run = await destination.get_agent_run(agent_run_id=run.agent_run_id)
                    observed_checkpoint = await destination.latest_checkpoint(
                        agent_run_id=run.agent_run_id,
                        include_interrupted=True,
                    )
                    observed_values = await destination.list_model_interactions(
                        agent_run_id=run.agent_run_id
                    )
                    observed_interactions = tuple(
                        value
                        for value in observed_values
                        if isinstance(value, ModelInteractionRecord)
                    )
                    if len(observed_interactions) != len(observed_values):
                        return CommitObservation(
                            DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                            error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                        )
                    observed_by_sequence = {value.model_request_seq: value for value in observed_interactions}
                    observed_interactions = tuple(
                        observed_by_sequence[value.model_request_seq]
                        for value in source_interactions if value.model_request_seq in observed_by_sequence
                    )
                    observed_resolved = tuple(
                        await destination.resolve_model_interactions(observed_interactions)
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                checkpoint_matches = _relocated_checkpoint_matches(
                    target,
                    checkpoint,
                    observed_checkpoint,
                )
                if (
                    observed_run == run
                    and checkpoint_matches
                    and tuple(
                        _interaction_semantic_header(value)
                        for value in observed_interactions
                    )
                    == tuple(
                        _interaction_semantic_header(value)
                        for value in source_interactions
                    )
                    and observed_resolved == source_resolved
                ):
                    return CommitObservation(DurableCommitState.COMMITTED)
                return CommitObservation(DurableCommitState.NOT_COMMITTED)

            await self._settle_durability_flight(flight, operation, readback)
            return

    async def prepare_execution_terminal_seal(
        self, *, execution_id: str, agent_run_ids: Sequence[str], binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        return await self._settle_observation_boundary(
            self._prepare_execution_terminal_seal(
                execution_id=execution_id, agent_run_ids=agent_run_ids,
                binding_digest=binding_digest,
            ),
            cancellation_cleanup=self.discard_execution_terminal_seal,
        )

    async def _prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        agent_run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        await self._ensure_business()
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        ordered_agent_run_ids = tuple(sorted(dict.fromkeys(agent_run_ids)))
        for agent_run_id in ordered_agent_run_ids:
            await self._stop_scheduled_projection(agent_run_id)
        if not ordered_agent_run_ids:
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
            for agent_run_id in ordered_agent_run_ids:
                while True:
                    archive_run: AgentRunRecord | None = None
                    captured_seal: _LocalExecutionTerminalSeal | None = None
                    needs_archive_run = False
                    async with self._history_lock.hold(agent_run_id):
                        existing = self._durability_flights.get(agent_run_id)
                        if existing is not None:
                            completion = existing.completion
                        else:
                            seal = self._terminal_seals.get(agent_run_id)
                            if seal is None:
                                seal = _LocalExecutionTerminalSeal(
                                    execution_id,
                                    terminal_attempt_token,
                                )
                                self._terminal_seals[agent_run_id] = seal
                                self._install_durability_flight_locked(
                                    agent_run_id,
                                    _AgentRunDurabilityKind.TERMINAL,
                                    token=terminal_attempt_token,
                                )
                                installed.append(agent_run_id)
                            elif seal.execution_id != execution_id or seal.token != terminal_attempt_token:
                                raise AIError(ErrorCode.STORAGE_CONFLICT)
                            captured_seal = seal
                            run = self._staging.get_agent_run_local(agent_run_id)
                            if run is None:
                                needs_archive_run = True
                            else:
                                projection = self._capture_projection_checkpoint_locked(agent_run_id)
                                captured.append((seal, projection))
                                break
                    if needs_archive_run:
                        archive_run = await archive.get_agent_run(agent_run_id=agent_run_id)
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
                tuple(projection.run.agent_run_id for _seal, projection in captured)
            )
            for _seal, projection in captured:
                durable_head = durable_heads.get(projection.run.agent_run_id)
                if durable_head is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                observation = None
                if projection.transcript is not None:
                    observation = await archive.transcript_repository.prepare_observation(
                        projection.run.agent_run_id,
                        projection.transcript.messages,
                        first_message_index=projection.base_message_index,
                        pending=projection.transcript.pending,
                        pending_keys=projection.transcript.pending_keys,
                    )
                batch = await archive.prepare_checkpoints_after_seal(
                    projection.run,
                    projection.checkpoints,
                    observed_message_count=None if observation is None else observation.target_message_count,
                )
                local_base = 0
                local_count = (projection.target_message_index if projection.transcript is not None
                               else batch.target_transcript_message_count)
                prepared.append(
                    PreparedExecutionProjection(
                        projection.run,
                        projection.events,
                        batch.checkpoints,
                        projection.base_event_offset,
                        projection.base_checkpoint_offset,
                        durable_head.event_count
                        + projection.target_event_offset
                        - projection.base_event_offset,
                        durable_head.checkpoint_count
                        + projection.target_checkpoint_offset
                        - projection.base_checkpoint_offset,
                        batch.target_transcript_message_count,
                        "empty"
                        if not batch.checkpoints
                        and durable_head.projection_digest == "empty"
                        else (
                            batch.checkpoints[-1].projection.digest
                            if batch.checkpoints
                            else durable_head.projection_digest
                        ),
                        tuple(
                            await archive.prepare_interactions(
                                projection.run,
                                projection.interactions,
                                lambda digest,
                                agent_run_id=projection.run.agent_run_id: self._staging.staged_payload(
                                    agent_run_id,
                                    digest,
                                ),
                                local_message_base=local_base,
                                local_message_count=local_count,
                            )
                        ),
                        durable_head.interaction_count
                        ,
                        max(durable_head.interaction_count, projection.target_interaction_offset),
                        observation,
                        self._execution_producers.get(projection.run.agent_run_id, (None, None))[1],
                    )
                )
            plan = ExecutionTerminalSealPlan(
                execution_id,
                binding_digest,
                tuple(prepared),
                tuple(
                    (projection.run.agent_run_id, seal.token)
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
            for agent_run_id in installed:
                await self._release_terminal_seal_if_owned(
                    agent_run_id,
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
            await self._settle_committed_terminal_projection(
                plan,
                projection,
                require_owned=True,
            )
        _logger.info(
            "execution terminal seal finalized: execution=%s runs=%s",
            plan.execution_id,
            len(plan.projections),
        )

    async def reconcile_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        for projection in plan.projections:
            await self._settle_committed_terminal_projection(
                plan,
                projection,
                require_owned=False,
            )
        _logger.warning(
            "execution terminal seal reconciled after local finalization failure: "
            "execution=%s runs=%s",
            plan.execution_id,
            len(plan.projections),
        )

    async def _settle_committed_terminal_projection(
        self,
        plan: ExecutionTerminalSealPlan,
        projection: PreparedExecutionProjection,
        *,
        require_owned: bool,
    ) -> None:
        agent_run_id = projection.run.agent_run_id
        token = plan.token_for(agent_run_id)
        completion: asyncio.Future[None] | None = None
        async with self._history_lock.hold(agent_run_id):
            seal = self._terminal_seals.get(agent_run_id)
            flight = self._durability_flights.get(agent_run_id)
            if require_owned and (seal is None or flight is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (seal is None) != (flight is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if seal is not None and (
                seal.execution_id != plan.execution_id
                or seal.token != token
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if flight is not None and (
                flight.kind is not _AgentRunDurabilityKind.TERMINAL
                or flight.token != token
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if flight is not None:
                del self._durability_flights[agent_run_id]
                completion = flight.completion
            if seal is not None:
                del self._terminal_seals[agent_run_id]
            offset = self._projection_offsets.setdefault(
                agent_run_id,
                _ProjectionOffset(),
            )
            offset.events = max(offset.events, projection.target_event_offset)
            offset.checkpoints = max(offset.checkpoints, projection.target_checkpoint_offset)
            offset.transcript_messages = max(
                offset.transcript_messages,
                projection.target_transcript_message_count,
            )
            offset.interactions = max(
                offset.interactions,
                projection.target_interaction_offset,
            )
            self._projection_dirty.discard(agent_run_id)
        self._publish_model_interaction_change(agent_run_id)
        if completion is not None and not completion.done():
            completion.set_result(None)

    async def discard_execution_terminal_seal(
        self,
        plan: ExecutionTerminalSealPlan,
    ) -> None:
        for agent_run_id in dict.fromkeys(
            projection.run.agent_run_id for projection in plan.projections
        ):
            await self._release_terminal_seal_if_owned(
                agent_run_id,
                execution_id=plan.execution_id,
                token=plan.token_for(agent_run_id),
            )

    async def _release_terminal_seal_if_owned(
        self,
        agent_run_id: str,
        *,
        execution_id: str,
        token: str,
    ) -> None:
        """Release one run's terminal seal only when this attempt owns it."""
        completion: asyncio.Future[None] | None = None
        async with self._history_lock.hold(agent_run_id):
            seal = self._terminal_seals.get(agent_run_id)
            if seal is None:
                return
            if seal.execution_id != execution_id or seal.token != token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            flight = self._durability_flights.get(agent_run_id)
            if flight is not None:
                if (
                    flight.kind is not _AgentRunDurabilityKind.TERMINAL
                    or flight.token != token
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                del self._durability_flights[agent_run_id]
                completion = flight.completion
            del self._terminal_seals[agent_run_id]
        if completion is not None and not completion.done():
            completion.set_result(None)

    def _capture_projection_checkpoint_locked(
        self,
        agent_run_id: str,
    ) -> ExecutionProjectionBatch:
        offset = self._projection_offsets.setdefault(agent_run_id, _ProjectionOffset())
        projection = self._staging.capture_projection_local(agent_run_id, offset)
        if projection is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return projection

    def _ensure_run_mutable(self, agent_run_id: str) -> None:
        if agent_run_id in self._terminal_seals:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def capture_execution_projection(
        self,
        agent_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _AgentRunProjectionFlight] | None":
        """CAPTURE: checkpoint staged state under the run lock with no durable I/O."""
        await self._ensure_business()
        while True:
            completion: asyncio.Future[None] | None = None
            async with self._history_lock.hold(agent_run_id):
                existing = self._durability_flights.get(agent_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    self._ensure_run_mutable(agent_run_id)
                    if agent_run_id not in self._projection_dirty:
                        return None
                    offset = self._projection_offsets.setdefault(
                        agent_run_id,
                        _ProjectionOffset(),
                    )
                    projection = self._staging.capture_projection_local(
                        agent_run_id,
                        offset,
                    )
                    if projection is None:
                        return None
                    flight = self._install_durability_flight_locked(
                        agent_run_id,
                        _AgentRunDurabilityKind.PROJECTION,
                    )
                    captured = CapturedExecutionProjection(
                        projection.run,
                        projection.events,
                        projection.checkpoints,
                        projection.base_event_offset,
                        projection.base_checkpoint_offset,
                        projection.target_event_offset,
                        projection.target_checkpoint_offset,
                        projection.interactions,
                        projection.base_interaction_offset,
                        projection.target_interaction_offset,
                        projection.transcript,
                        projection.base_message_index,
                        projection.target_message_index,
                        self._execution_producers.get(agent_run_id, (None, None))[1],
                    )
                    _logger.debug(
                        "projection flight captured: agent_run=%s token=%s "
                        "events=%s checkpoints=%s",
                        agent_run_id,
                        flight.token,
                        len(captured.events),
                        len(captured.checkpoints),
                    )
                    return captured, flight
            await asyncio.shield(completion)

    async def wait_projection_flight(self, agent_run_id: str) -> None:
        """Wait for an active flight without holding the run lock."""
        await self._ensure_business()
        while True:
            async with self._history_lock.hold(agent_run_id):
                existing = self._durability_flights.get(agent_run_id)
                if existing is None:
                    return
                completion = existing.completion
            await asyncio.shield(completion)

    async def abandon_execution_projection(
        self,
        flight: _AgentRunProjectionFlight,
    ) -> None:
        """Remove a flight after a definitely-not-committed outcome."""
        async with self._history_lock.hold(flight.agent_run_id):
            if self._durability_flights.get(flight.agent_run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.agent_run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "projection flight abandoned: agent_run=%s token=%s",
            flight.agent_run_id,
            flight.token,
        )

    async def finalize_execution_projection(
        self,
        flight: _AgentRunProjectionFlight,
        captured: CapturedExecutionProjection,
        *,
        target_transcript_message_count: int | None = None,
    ) -> None:
        """FINALIZE: advance offsets and clear dirty state after durable success."""
        async with self._history_lock.hold(flight.agent_run_id):
            if self._durability_flights.get(flight.agent_run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.agent_run_id]
            offset = self._projection_offsets.setdefault(
                flight.agent_run_id,
                _ProjectionOffset(),
            )
            offset.events = max(offset.events, captured.target_event_offset)
            offset.checkpoints = max(offset.checkpoints, captured.target_checkpoint_offset)
            offset.interactions = max(
                offset.interactions,
                captured.target_interaction_offset,
            )
            self._staging.acknowledge_interactions_local(
                flight.agent_run_id, captured.interactions,
            )
            if target_transcript_message_count is not None:
                offset.transcript_messages = max(
                    offset.transcript_messages,
                    target_transcript_message_count,
                )
            remaining = self._staging.capture_projection_local(flight.agent_run_id, offset)
            pending_changed = (
                remaining is not None and remaining.transcript is not None
                and (captured.transcript is None
                     or remaining.transcript.pending != captured.transcript.pending
                     or remaining.transcript.pending_keys != captured.transcript.pending_keys)
            )
            if remaining is not None and (
                remaining.events or remaining.checkpoints or remaining.interactions
                or remaining.transcript is not None and remaining.transcript.messages
                or pending_changed
            ):
                self._projection_dirty.add(flight.agent_run_id)
            else:
                self._projection_dirty.discard(flight.agent_run_id)
                if offset.observation_task is None:
                    offset.observation_execution_id = None
                    offset.observation_producer_generation = None
        self._publish_model_interaction_change(flight.agent_run_id)
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "projection flight finalized: agent_run=%s token=%s events=%s checkpoints=%s",
            flight.agent_run_id,
            flight.token,
            captured.target_event_offset,
            captured.target_checkpoint_offset,
        )

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _AgentRunProjectionFlight,
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
                    captured.checkpoints,
                    captured.interactions,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    stored_run = await archive.get_agent_run(agent_run_id=captured.run.agent_run_id)
                    stored_events = await archive.list_events(
                        agent_run_id=captured.run.agent_run_id
                    )
                    stored_checkpoint = await archive.latest_checkpoint(
                        agent_run_id=captured.run.agent_run_id,
                        include_interrupted=True,
                    )
                    stored_interactions = await archive.list_model_interactions(
                        agent_run_id=captured.run.agent_run_id
                    )
                except AIError as error:
                    return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
                if stored_run != captured.run:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.events and tuple(stored_events[-len(captured.events) :]) != captured.events:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
                if captured.checkpoints and stored_checkpoint != captured.checkpoints[-1]:
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
                "projection commit unresolved; flight retained: agent_run=%s token=%s",
                flight.agent_run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        if not captured.events and not captured.checkpoints and not captured.interactions and captured.transcript is None:
            await self.finalize_execution_projection(flight, captured)
            return
        started = monotonic()
        try:
            observation = None
            if captured.transcript is not None:
                observation = await archive.transcript_repository.prepare_observation(
                    captured.run.agent_run_id,
                    captured.transcript.messages,
                    first_message_index=captured.base_message_index,
                    pending=captured.transcript.pending,
                    pending_keys=captured.transcript.pending_keys,
                )
            prepared = await archive.prepare_checkpoints(
                captured.run,
                captured.checkpoints,
                observed_message_count=None if observation is None else observation.target_message_count,
            )
            local_count = (captured.target_message_index if captured.transcript is not None
                           else prepared.target_transcript_message_count)
            interactions = await archive.prepare_interactions(
                captured.run,
                captured.interactions,
                lambda digest: self._staging.staged_payload(captured.run.agent_run_id, digest),
                local_message_base=0,
                local_message_count=local_count,
            )
        except BaseException:
            await self.abandon_execution_projection(flight)
            raise
        durable_head = await archive.execution_history_head_record(captured.run.agent_run_id)
        expected_head = ExecutionRunSealHead(
            captured.run.agent_run_id,
            durable_head.event_count + len(captured.events),
            durable_head.checkpoint_count + len(prepared.checkpoints),
            prepared.target_transcript_message_count,
            prepared.checkpoints[-1].projection.digest if prepared.checkpoints else durable_head.projection_digest,
            max(durable_head.interaction_count, captured.target_interaction_offset),
        )

        async def operation() -> ExecutionRunSealHead:
            async def commit(group: StateGroupTransaction) -> ExecutionRunSealHead:
                await archive.sync_prepared_projection(
                    captured.run, events=captured.events, checkpoints=prepared.checkpoints,
                    interactions=interactions, observation=observation,
                    producer_generation=captured.producer_generation, execution_id=execution_id,
                )
                head = await archive.execution_history_head_record(captured.run.agent_run_id)
                if (
                    head.event_count != expected_head.event_count
                    or head.checkpoint_count != expected_head.checkpoint_count
                    or head.transcript_message_count != expected_head.transcript_message_count
                    or head.interaction_count != expected_head.interaction_count
                    or prepared.checkpoints and head.projection_digest != expected_head.projection_digest
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return head

            return await archive.state_store.storage_group.mutate((archive.state_store,), commit)

        async def readback() -> CommitObservation[ExecutionRunSealHead]:
            try:
                head = await archive.execution_history_head_record(
                    captured.run.agent_run_id
                )
            except AIError as error:
                if error.code is ErrorCode.STORAGE_INTEGRITY_ERROR:
                    return CommitObservation(
                        DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                        error=error,
                    )
                return CommitObservation(DurableCommitState.UNRESOLVED, error=error)
            if head.event_count < expected_head.event_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.checkpoint_count < expected_head.checkpoint_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.transcript_message_count < expected_head.transcript_message_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if head.interaction_count < expected_head.interaction_count:
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if captured.events:
                observed_events = await archive.list_event_range(
                    agent_run_id=captured.run.agent_run_id,
                    start=durable_head.event_count,
                    end=expected_head.event_count,
                )
                if tuple(observed_events) != captured.events:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if not await archive.verify_checkpoint_range(
                agent_run_id=captured.run.agent_run_id,
                after_sequence=durable_head.checkpoint_count,
                checkpoints=prepared.checkpoints,
            ):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if interactions:
                try:
                    for expected in interactions:
                        observed = await archive.list_model_interactions(
                            agent_run_id=captured.run.agent_run_id,
                            after_model_request_seq=expected.model_request_seq - 1,
                            limit=1,
                        )
                        if not observed or not isinstance(observed[0], ModelInteractionRecord):
                            return CommitObservation(DurableCommitState.NOT_COMMITTED)
                        expected.validate_successor(observed[0])
                except AIError as error:
                    return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
            if observation is not None and not await archive.transcript_repository.verify_observation(observation):
                return CommitObservation(DurableCommitState.NOT_COMMITTED)
            if prepared.checkpoints and head.checkpoint_count == expected_head.checkpoint_count and head.projection_digest != expected_head.projection_digest:
                return CommitObservation(
                    DurableCommitState.PARTIAL_INTEGRITY_ERROR,
                    error=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR),
                )
            return CommitObservation(
                DurableCommitState.COMMITTED,
                value=ExecutionRunSealHead(
                    captured.run.agent_run_id,
                    head.event_count,
                    head.checkpoint_count,
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
                "projection commit unresolved; flight retained: agent_run=%s token=%s",
                flight.agent_run_id,
                flight.token,
            )
            unknown = AIError(
                ErrorCode.STORAGE_COMMIT_UNKNOWN,
                "projection commit outcome is unresolved",
            )
            await self._fence_durability_flight(flight, unknown)
            raise unknown from result.error
        _logger.debug(
            "step projection flushed: domain=%s backend=%s agent_run=%s "
            "events=%s checkpoints=%s duration_ms=%.3f",
            RuntimeDomain.EXECUTION.value,
            type(archive).__name__,
            flight.agent_run_id,
            len(captured.events),
            len(captured.checkpoints),
            (monotonic() - started) * 1000,
        )

    async def flush_execution_projection(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None = None, deferred: bool = False,
    ) -> None:
        operation = self._flush_execution_projection(
            agent_run_id, execution_id=execution_id,
            producer_generation=producer_generation, deferred=deferred,
        )
        if deferred:
            await operation
        else:
            await self._settle_observation_boundary(operation)

    async def _flush_execution_projection(
        self,
        agent_run_id: str,
        *,
        execution_id: str,
        producer_generation: int | None = None,
        deferred: bool = False,
    ) -> None:
        if RuntimeDomain.EXECUTION not in self._archives:
            await self._ensure_business()
            # A transient route is observable only through its local staging
            # owner; there is no archive projection to commit or acknowledge.
            self._projection_dirty.discard(agent_run_id)
            self._publish_model_interaction_change(agent_run_id)
            return
        offset = self._projection_offsets.get(agent_run_id)
        if (agent_run_id in self._projection_dirty and offset is not None
                and offset.observation_execution_id is not None):
            if (execution_id != offset.observation_execution_id
                    or producer_generation is not None
                    and producer_generation != offset.observation_producer_generation):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            producer_generation = offset.observation_producer_generation
        elif producer_generation is None:
            producer_generation = self._execution_producers.get(agent_run_id, (None, None))[1]
        if deferred:
            await self._defer_execution_projection(
                agent_run_id, execution_id=execution_id,
                producer_generation=producer_generation,
            )
            return
        await self._stop_scheduled_projection(agent_run_id)
        captured = await self.capture_execution_projection(
            agent_run_id,
        )
        if captured is None:
            return
        projection, flight = captured
        if producer_generation is not None:
            projection = replace(projection, producer_generation=producer_generation)
        await self.commit_captured_execution_projection(
            projection,
            flight,
            execution_id=execution_id,
        )

    async def _defer_execution_projection(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None,
    ) -> None:
        await self._ensure_business()
        async with self._history_lock.hold(agent_run_id):
            self._ensure_run_mutable(agent_run_id)
            offset = self._projection_offsets.setdefault(agent_run_id, _ProjectionOffset())
            task = offset.observation_task
            if task is not None and task.done():
                if agent_run_id not in self._durability_flights:
                    offset.observation_task = None
                    offset.observation_boundaries = 0
                task.result()
            if agent_run_id not in self._projection_dirty:
                return
            schedule = offset.observation_boundaries == 0
            offset.observation_boundaries += 1
            offset.observation_execution_id = execution_id
            offset.observation_producer_generation = producer_generation
            pressure = offset.observation_boundaries >= _OBSERVATION_FLUSH_BOUNDARIES
        if pressure:
            await self.flush_execution_projection(
                agent_run_id, execution_id=execution_id,
                producer_generation=producer_generation,
            )
        elif schedule:
            self._schedule_observation_flush()

    def _schedule_observation_flush(self) -> None:
        # Tasks inherit ContextVars; callers leave their run lock before this.
        self._observation_wake.set()
        task = self._observation_task
        if task is not None:
            if not task.done():
                return
            task.result()
        if self._observation_started_at is None:
            self._observation_started_at = monotonic()
        self._observation_task = asyncio.create_task(
            self._run_observation_scheduler(), name="history-observation-scheduler",
        )
        self._track_observation_task(self._observation_task)

    def _track_observation_task(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.add(task)

        def completed(value: asyncio.Task[None]) -> None:
            self._background_tasks.discard(value)
            self._observation_wake.set()
            if not value.cancelled():
                value.exception()

        task.add_done_callback(completed)

    async def _run_observation_scheduler(self) -> None:
        while True:
            self._observation_wake.clear()
            queued = tuple(
                agent_run_id for agent_run_id in self._projection_dirty
                if (offset := self._projection_offsets.get(agent_run_id)) is not None
                and offset.observation_boundaries
                and (offset.observation_task is None or not offset.observation_task.done())
            )
            if not queued:
                self._observation_started_at = None
                return
            if self._observation_started_at is None:
                self._observation_started_at = monotonic()
            if all(self._projection_offsets[run_id].observation_task is not None for run_id in queued):
                await self._observation_wake.wait()
                continue
            remaining = max(
                0.0, self._observation_started_at + _OBSERVATION_FLUSH_SECONDS - monotonic(),
            )
            try:
                await asyncio.wait_for(self._observation_wake.wait(), remaining)
            except asyncio.TimeoutError:
                pass
            if self._observation_wake.is_set():
                continue
            self._observation_started_at = None
            for agent_run_id in queued:
                async with self._history_lock.hold(agent_run_id):
                    offset = self._projection_offsets.get(agent_run_id)
                    if (offset is None or not offset.observation_boundaries
                            or offset.observation_task is not None
                            or agent_run_id not in self._projection_dirty):
                        continue
                    execution_id = offset.observation_execution_id
                    if execution_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    offset.observation_boundaries = 0
                    generation = offset.observation_producer_generation
                task = asyncio.create_task(
                    self._run_scheduled_projection(
                        agent_run_id, offset, execution_id=execution_id,
                        producer_generation=generation,
                    ),
                    name=f"history-observation-commit-{agent_run_id}",
                )
                offset.observation_task = task
                self._track_observation_task(task)

    async def _run_scheduled_projection(
        self, agent_run_id: str, offset: _ProjectionOffset, *,
        execution_id: str, producer_generation: int | None,
    ) -> None:
        await self._flush_execution_projection(
            agent_run_id, execution_id=execution_id,
            producer_generation=producer_generation,
        )
        async with self._history_lock.hold(agent_run_id):
            if offset.observation_task is not asyncio.current_task():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            offset.observation_task = None
            dirty = agent_run_id in self._projection_dirty
            if dirty:
                offset.observation_boundaries = max(1, offset.observation_boundaries)
            else:
                offset.observation_boundaries = 0
                offset.observation_execution_id = None
                offset.observation_producer_generation = None
        if dirty:
            self._schedule_observation_flush()

    async def _stop_scheduled_projection(self, agent_run_id: str) -> None:
        """Remove this run from the timer and settle only its captured commit."""
        async with self._history_lock.hold(agent_run_id):
            offset = self._projection_offsets.get(agent_run_id)
            task = None if offset is None else offset.observation_task
            if offset is None or task is asyncio.current_task():
                return
            offset.observation_boundaries = 0
            self._observation_wake.set()
            if task is None:
                return
        try:
            await asyncio.shield(task)
        except Exception:
            async with self._history_lock.hold(agent_run_id):
                if agent_run_id in self._durability_flights:
                    raise
                offset.observation_task = None
                offset.observation_boundaries = 0
            _logger.warning(
                "observation batch failed before commit; required barrier will retry: agent_run=%s",
                agent_run_id,
            )

    async def _settle_observation_boundary(
        self, operation: Awaitable[_BoundaryResult], *,
        cancellation_cleanup: Callable[[_BoundaryResult], Awaitable[None]] | None = None,
    ) -> _BoundaryResult:
        """Keep cancellation outside the whole checkpoint, drain, or release."""
        async def run() -> _BoundaryResult:
            return await operation

        value, error, cancelled = await _await_owned_task(
            asyncio.create_task(run()), self._background_tasks,
        )
        if error is not None:
            raise error
        result = cast(_BoundaryResult, value)
        if cancelled:
            if cancellation_cleanup is not None:
                await self._settle_observation_boundary(cancellation_cleanup(result))
            raise asyncio.CancelledError
        return result

    async def flush_dirty_execution_projections(self, *, execution_id: str) -> None:
        for agent_run_id in tuple(self._projection_dirty):
            await self.flush_execution_projection(agent_run_id, execution_id=execution_id)


    async def verify_terminal_attempts(
        self, *, candidate_agent_run_ids: tuple[str, ...], required_agent_run_id: str | None
    ) -> None:
        for agent_run_id in dict.fromkeys(candidate_agent_run_ids):
            if required_agent_run_id != agent_run_id:
                continue
            checkpoint = await self._staging.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
            if checkpoint is None:
                for archive in self._archives.values():
                    checkpoint = await archive.latest_checkpoint(
                        agent_run_id=agent_run_id,
                        include_interrupted=True,
                    )
                    if checkpoint is not None:
                        break
            if checkpoint is None or checkpoint.state != "complete":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def release_staging_many(
        self, *, candidate_agent_run_ids: tuple[str, ...], execution_id: str | None = None,
    ) -> None:
        await self._settle_observation_boundary(self._release_staging_many(
            candidate_agent_run_ids=candidate_agent_run_ids, execution_id=execution_id,
        ))

    async def discard_revoked_producer_staging(
        self, agent_run_id: str, *, execution_id: str, producer_generation: int,
    ) -> None:
        """Release a quiescent producer's uncommitted tail after revocation proof."""
        await self._settle_observation_boundary(self._discard_revoked_producer_staging(
            agent_run_id, execution_id=execution_id, producer_generation=producer_generation,
        ))

    async def _discard_revoked_producer_staging(
        self, agent_run_id: str, *, execution_id: str, producer_generation: int,
    ) -> None:
        await self._ensure_business()
        async with self._history_lock.hold(agent_run_id):
            if (agent_run_id not in self._projection_dirty
                    and agent_run_id not in self._execution_producers):
                return
            if self._execution_producers.get(agent_run_id) != (execution_id, producer_generation):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        await self._stop_scheduled_projection(agent_run_id)
        await self.wait_projection_flight(agent_run_id)
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, StateStepArchive):
            if archive is None and agent_run_id not in self._projection_dirty:
                return
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        await archive.require_revoked_execution_producer(
            execution_id=execution_id, producer_generation=producer_generation,
        )
        async with self._history_lock.hold(agent_run_id):
            if (self._execution_producers.get(agent_run_id) != (execution_id, producer_generation)
                    or agent_run_id in self._durability_flights):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            self._projection_dirty.discard(agent_run_id)
        await self._release_staging_many(
            candidate_agent_run_ids=(agent_run_id,), execution_id=execution_id,
        )
        _logger.info("revoked producer staging released: agent_run=%s execution=%s generation=%s",
                     agent_run_id, execution_id, producer_generation)

    async def _release_staging_many(
        self,
        *,
        candidate_agent_run_ids: tuple[str, ...],
        execution_id: "str | None" = None,
    ) -> None:
        for agent_run_id in dict.fromkeys(candidate_agent_run_ids):
            await self._stop_scheduled_projection(agent_run_id)
            while True:
                completion: asyncio.Future[None] | None = None
                seal_owner: str | None = None
                seal_token: str | None = None
                async with self._history_lock.hold(agent_run_id):
                    existing = self._durability_flights.get(agent_run_id)
                    if existing is not None:
                        completion = existing.completion
                    else:
                        if agent_run_id in self._projection_dirty:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        self._publish_model_interaction_change(agent_run_id)
                        self._staging.release_agent_run_local(agent_run_id)
                        self._execution_producers.pop(agent_run_id, None)
                        self._projection_offsets.pop(agent_run_id, None)
                        self._projection_dirty.discard(agent_run_id)
                        for archive in self._archives.values():
                            if isinstance(archive, StateStepArchive):
                                archive.release_runtime_cache(agent_run_id)
                        seal = self._terminal_seals.get(agent_run_id)
                        if seal is not None and seal.execution_id == execution_id:
                            seal_owner = seal.execution_id
                            seal_token = seal.token
                if completion is None:
                    break
                await asyncio.shield(completion)
            if seal_owner is not None and seal_token is not None:
                await self._release_terminal_seal_if_owned(
                    agent_run_id,
                    execution_id=seal_owner,
                    token=seal_token,
                )
                _logger.warning(
                    "terminal seal discarded on staging release: agent_run=%s "
                    "execution=%s",
                    agent_run_id,
                    seal_owner,
                )

    async def release_archive(
        self, runtime_domain: RuntimeDomain, agent_run_id: str, *,
        execution_id: str | None = None,
    ) -> None:
        await self._settle_observation_boundary(self._release_archive(
            runtime_domain, agent_run_id, execution_id=execution_id,
        ))

    async def _release_archive(
        self,
        runtime_domain: RuntimeDomain,
        agent_run_id: str,
        *,
        execution_id: str | None = None,
    ) -> None:
        archive = self._archives.get(runtime_domain)
        if archive is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        await self._stop_scheduled_projection(agent_run_id)
        while True:
            completion: asyncio.Future[None] | None = None
            flight: _AgentRunDurabilityFlight | None = None
            async with self._history_lock.hold(agent_run_id):
                existing = self._durability_flights.get(agent_run_id)
                if existing is not None:
                    completion = existing.completion
                else:
                    if runtime_domain is RuntimeDomain.EXECUTION:
                        self._ensure_run_mutable(agent_run_id)
                    self._projection_offsets.pop(agent_run_id, None)
                    self._projection_dirty.discard(agent_run_id)
                    flight = self._install_durability_flight_locked(
                        agent_run_id,
                        _AgentRunDurabilityKind.RELEASE,
                    )
            if completion is not None:
                await asyncio.shield(completion)
                continue
            if flight is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            async def operation() -> None:
                await archive.release_agent_run(
                    agent_run_id,
                    execution_id=execution_id,
                )

            async def readback() -> CommitObservation[None]:
                try:
                    observed = await archive.get_agent_run(agent_run_id=agent_run_id)
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
        agent_run_id: str,
        kind: _AgentRunDurabilityKind,
        *,
        token: str | None = None,
    ) -> _AgentRunDurabilityFlight:
        existing = self._durability_flights.get(agent_run_id)
        if existing is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        flight = _AgentRunDurabilityFlight(
            agent_run_id,
            token or uuid4().hex,
            kind,
            asyncio.get_running_loop().create_future(),
        )
        self._durability_flights[agent_run_id] = flight
        _logger.debug(
            "durability flight captured: agent_run=%s token=%s kind=%s",
            agent_run_id,
            flight.token,
            kind.value,
        )
        return flight

    async def _finalize_durability_flight(
        self,
        flight: _AgentRunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.agent_run_id):
            if self._durability_flights.get(flight.agent_run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.agent_run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.debug(
            "durability flight finalized: agent_run=%s token=%s kind=%s",
            flight.agent_run_id,
            flight.token,
            flight.kind.value,
        )

    async def _abandon_durability_flight(
        self,
        flight: _AgentRunDurabilityFlight,
    ) -> None:
        async with self._history_lock.hold(flight.agent_run_id):
            if self._durability_flights.get(flight.agent_run_id) is not flight:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            del self._durability_flights[flight.agent_run_id]
        if not flight.completion.done():
            flight.completion.set_result(None)
        _logger.info(
            "durability flight abandoned: agent_run=%s token=%s kind=%s",
            flight.agent_run_id,
            flight.token,
            flight.kind.value,
        )

    async def _fence_durability_flight(
        self,
        flight: _AgentRunDurabilityFlight,
        error: AIError,
    ) -> None:
        async with self._history_lock.hold(flight.agent_run_id):
            current = self._durability_flights.get(flight.agent_run_id)
            if current is not flight or current.token != flight.token:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            completion = flight.completion
        if not completion.done():
            completion.set_exception(error)

            def consume(future: asyncio.Future[None]) -> None:
                future.exception()

            completion.add_done_callback(consume)
        _logger.error(
            "durability flight fenced: agent_run=%s token=%s kind=%s code=%s",
            flight.agent_run_id,
            flight.token,
            flight.kind.value,
            error.code.value,
        )

    async def _settle_durability_flight(
        self,
        flight: _AgentRunDurabilityFlight,
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
            "durability flight unresolved: agent_run=%s token=%s kind=%s",
            flight.agent_run_id,
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
        await self._settle_observation_boundary(self._preflight_close())

    async def _preflight_close(self) -> None:
        for agent_run_id in tuple(self._projection_offsets):
            await self._stop_scheduled_projection(agent_run_id)
        self._observation_wake.set()
        if self._observation_task is not None:
            await asyncio.shield(self._observation_task)
        for agent_run_id in tuple(self._projection_dirty):
            offset = self._projection_offsets.get(agent_run_id)
            if offset is not None and offset.observation_execution_id is not None:
                await self.flush_execution_projection(
                    agent_run_id, execution_id=offset.observation_execution_id,
                    producer_generation=offset.observation_producer_generation,
                )
        pending_tasks = tuple(
            task for task in self._background_tasks
            if not task.done() and task is not asyncio.current_task()
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
        for subscriptions in tuple(self._model_interaction_subscriptions.values()):
            for subscription in tuple(subscriptions):
                await subscription.close()
        await self._staging.close()
        for archive in self._archives.values():
            await archive.close()
        self._initialized = False

    async def _ensure_business(self) -> None:
        if not self._initialized:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


def _relocated_checkpoint_matches(
    target: RuntimeDomain,
    source: AgentRunCheckpoint,
    observed: AgentRunCheckpoint | None,
) -> bool:
    if observed is None:
        return False
    if target is RuntimeDomain.CONVERSATION:
        return _conversation_relocated_checkpoint_matches(source, observed)
    return replace(
        observed,
        transcript_message_count_before=None,
    ) == replace(
        source,
        transcript_message_count_before=None,
    )


def _agent_run_identity(run: AgentRunRecord) -> tuple[object, ...]:
    return (
        run.agent_run_id,
        run.agent_conversation_id,
        run.parent_agent_run_id,
        run.agent_id,
        tuple(sorted(run.metadata.items())),
    )


def _interaction_semantic_header(
    interaction: ModelInteractionRecord,
) -> tuple[object, ...]:
    return (
        interaction.agent_run_id,
        interaction.step_index,
        interaction.model_request_seq,
        interaction.purpose,
        interaction.output_retry_index,
        tuple(sorted(interaction.model.items())),
        interaction.status,
        interaction.error_code,
        interaction.duration_ns,
        interaction.usage,
        interaction.attachments,
    )


def _interaction_local_range(
    checkpoints: Sequence[AgentRunCheckpoint],
    target_transcript_message_count: int,
    *,
    fallback_local_count: int = 0,
) -> tuple[int, int]:
    if (
        target_transcript_message_count < 0
        or fallback_local_count < 0
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    local_count = (
        len(checkpoints[-1].messages)
        if checkpoints
        else fallback_local_count
    )
    local_base = target_transcript_message_count - local_count
    if local_base < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return local_base, local_count


__all__ = [
    "ExecutionProjectionBatch",
    "ExecutionTerminalSealPlan",
    "InMemoryStepArchive",
    "LockOrderError",
    "PreparedExecutionProjection",
    "PreparedAgentRunCheckpoint",
    "PreparedAgentRunCheckpointBatch",
    "RuntimeAgentRunStore",
    "StagingAgentRunStore",
    "StateStepArchive",
]
