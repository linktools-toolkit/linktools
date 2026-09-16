#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-interaction extensions for Runtime StepStore implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage

from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._model_interaction import (
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    context_projection_to_durable,
    extend_prefix_digest,
    message_prefix_digest,
)
from ._contracts import (
    ContextProjection,
    ExecutionHistoryHeadRecord,
    ExecutionRunSealHead,
    InlineContextBlock,
    ModelInteractionRecord,
    RuntimePayloadRef,
    TranscriptSpanRef,
)
from ._plan import RuntimeDomain
from ._step_contracts import ContinuableSnapshot, RunRecord, StepEvent
from ._steps import (
    CapturedExecutionProjection,
    ExecutionProjectionBatch,
    ExecutionTerminalSealPlan,
    InMemoryStepArchive,
    PreparedStepSnapshot,
    RuntimeStepStore,
    StagingStepStore,
    StateStepArchive,
    _ProjectionOffset,
    _decode_step,
    _encode_step,
    _insert_facts,
    _step_subject,
)
from ._store import FactQuery, StateTransaction, StoredFact, StoredRecord

if TYPE_CHECKING:
    from ._steps import _RunProjectionFlight


class ModelInteractionStagingStepStore(StagingStepStore):
    """Keep request sequence as the in-process interaction high-water."""

    def stage_model_interaction(self, interaction: object) -> None:
        self._ensure_open()
        _stage_interaction(self._interactions, interaction)

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        return _bounded_interactions(
            self._interactions.get(run_id, ()),
            after_request_sequence=after_request_sequence,
            limit=limit,
        )

    def capture_projection_local(
        self,
        run_id: str,
        offset: _ProjectionOffset,
    ) -> ExecutionProjectionBatch | None:
        return _capture_projection(self, run_id, offset)


class ModelInteractionInMemoryStepArchive(InMemoryStepArchive):
    """Volatile archive using the same resolved interaction contract as durable stores."""

    def stage_model_interaction(self, interaction: object) -> None:
        self._ensure_open()
        _stage_interaction(self._interactions, interaction)

    async def prepare_interactions(
        self,
        run: RunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = _validate_interaction_batch(run, interactions)
        if not values:
            return ()
        required_count = _required_source_message_count(values)
        if source_messages is not None and len(source_messages) >= required_count:
            prefix_messages = tuple(source_messages)
        else:
            snapshot = await self.latest_snapshot(
                run_id=run.run_id,
                include_interrupted=True,
            )
            prefix_messages = () if snapshot is None else tuple(snapshot.messages)
        _validate_interaction_sources(values, prefix_messages)

        def prepare_context(staged: StagedContextProjection) -> ContextProjection:
            return context_projection_to_durable(
                staged,
                owner_id=run.run_id,
                source_domain=self._runtime_domain,
                payload=payload,
            )

        return tuple(
            ModelInteractionRecord(
                staged.run_id,
                staged.step_index,
                staged.request_sequence,
                staged.purpose,
                staged.output_retry_index,
                staged.model,
                prepare_context(staged.request_context),
                RuntimePayloadRef(
                    StoredPayload.inline_bytes(payload(staged.request_envelope_digest)),
                    self._runtime_domain,
                ),
                None
                if staged.response_context is None
                else prepare_context(staged.response_context),
                staged.status,
                staged.error_code,
                staged.duration_ns,
                staged.usage,
            )
            for staged in values
        )

    async def sync_projection(
        self,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[ContinuableSnapshot],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None:
        current = {
            value.request_sequence: value
            for value in self._interactions.get(run.run_id, ())
            if isinstance(value, ModelInteractionRecord)
        }
        fresh: list[ModelInteractionRecord] = []
        for interaction in interactions:
            if not isinstance(interaction, ModelInteractionRecord):
                raise TypeError("model interaction is invalid")
            previous = current.get(interaction.request_sequence)
            if previous is None:
                fresh.append(interaction)
                current[interaction.request_sequence] = interaction
            elif previous != interaction:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await super().sync_projection(
            run,
            events=events,
            snapshots=snapshots,
            interactions=fresh,
            execution_id=execution_id,
        )

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        values = self._interactions.get(run_id, ())
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _bounded_interactions(
            values,
            after_request_sequence=after_request_sequence,
            limit=limit,
        )

    def capture_projection_local(
        self,
        run_id: str,
        offset: _ProjectionOffset,
    ) -> ExecutionProjectionBatch | None:
        return _capture_projection(self, run_id, offset)


class ModelInteractionStateStepArchive(StateStepArchive):
    """Durable archive that keys one immutable fact per logical model request."""

    async def resolve_model_interactions(
        self,
        interactions: Sequence[object],
    ) -> list[object]:
        values = tuple(interactions)
        if not values:
            return []
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise TypeError("model interaction is invalid")
        records = tuple(
            value for value in values if isinstance(value, ModelInteractionRecord)
        )
        if any(record.run_id != records[0].run_id for record in records):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await super().resolve_model_interactions(records)

    async def prepare_interactions(
        self,
        run: RunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = _validate_interaction_batch(run, interactions)
        if not values:
            return ()
        required_count = _required_source_message_count(values)
        if source_messages is not None and len(source_messages) >= required_count:
            prefix_messages = tuple(source_messages)
        else:
            prefix_messages = await self._history.load_messages(run.run_id)
        _validate_interaction_sources(values, prefix_messages)

        prepared_payloads: dict[str, RuntimePayloadRef] = {}

        async def prepare_payload(digest: str) -> RuntimePayloadRef:
            prepared = prepared_payloads.get(digest)
            if prepared is not None:
                return prepared
            prepared = await self._prepare_inline_payload(
                run.run_id,
                RuntimePayloadRef(
                    StoredPayload.inline_bytes(payload(digest)),
                    self._runtime_domain,
                ),
            )
            prepared_payloads[digest] = prepared
            return prepared

        async def prepare_context(
            staged: StagedContextProjection,
        ) -> ContextProjection:
            projection = context_projection_to_durable(
                staged,
                owner_id=run.run_id,
                source_domain=self._runtime_domain,
                payload=payload,
            )
            items = []
            for item in projection.items:
                if isinstance(item, TranscriptSpanRef):
                    items.append(item)
                    continue
                items.append(
                    InlineContextBlock(
                        await prepare_payload(item.content.payload.digest)
                    )
                )
            return ContextProjection(tuple(items))

        result: list[ModelInteractionRecord] = []
        for staged in values:
            result.append(
                ModelInteractionRecord(
                    staged.run_id,
                    staged.step_index,
                    staged.request_sequence,
                    staged.purpose,
                    staged.output_retry_index,
                    staged.model,
                    await prepare_context(staged.request_context),
                    await prepare_payload(staged.request_envelope_digest),
                    None
                    if staged.response_context is None
                    else await prepare_context(staged.response_context),
                    staged.status,
                    staged.error_code,
                    staged.duration_ns,
                    staged.usage,
                )
            )
        return tuple(result)

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        _validate_page(after_request_sequence, limit)
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    self._stream(run_id, "interaction"),
                    after_sequence=after_request_sequence,
                    limit=limit,
                )
            )
        )
        return [self._decode_interaction_fact(fact) for fact in values]

    def _decode_interaction_fact(self, fact: StoredFact) -> ModelInteractionRecord:
        value = _decode_step(fact.data)
        if (
            not isinstance(value, ModelInteractionRecord)
            or fact.sequence != value.request_sequence
            or fact.subject_digest != _step_subject(value)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        *,
        events: Sequence[StepEvent],
        snapshots: Sequence[PreparedStepSnapshot],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        supplied_guard = history_head_guard is not None
        guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            history_head_guard,
        )
        owner = self._run_key(run.run_id)
        run_existed = await transaction.get_record(owner) is not None
        await super()._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            snapshots=snapshots,
            interactions=(),
            execution_id=execution_id,
            history_head_guard=guard,
        )
        inserted = await self._sync_interactions_in_transaction(
            transaction,
            run,
            interactions,
            owner_already_guarded=bool(events or snapshots),
        )
        if guard is not None and not supplied_guard and (
            not run_existed or events or snapshots or inserted
        ):
            await self._advance_execution_history_head_in_transaction(
                transaction,
                guard,
            )

    async def _sync_interactions_in_transaction(
        self,
        transaction: StateTransaction,
        run: RunRecord,
        interactions: Sequence[ModelInteractionRecord],
        *,
        owner_already_guarded: bool,
    ) -> int:
        values = tuple(interactions)
        if not values:
            return 0
        sequences = tuple(value.request_sequence for value in values)
        if (
            any(value.run_id != run.run_id for value in values)
            or sequences != tuple(sorted(sequences))
            or len(set(sequences)) != len(sequences)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        sequence_key = self._sequence(run.run_id, "interaction")
        durable_count = (await transaction.get_sequences((sequence_key,))).get(
            sequence_key,
            0,
        )
        replay = tuple(
            value for value in values if value.request_sequence <= durable_count
        )
        fresh = tuple(
            value for value in values if value.request_sequence > durable_count
        )
        if replay:
            replay_sequences = tuple(value.request_sequence for value in replay)
            if replay_sequences != tuple(
                range(
                    replay_sequences[0],
                    replay_sequences[0] + len(replay_sequences),
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored = await transaction.list_facts(
                FactQuery(
                    self._stream(run.run_id, "interaction"),
                    after_sequence=replay_sequences[0] - 1,
                    limit=len(replay),
                )
            )
            if len(stored) != len(replay):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for fact, value in zip(stored, replay, strict=True):
                if (
                    fact.sequence != value.request_sequence
                    or fact.subject_digest != _step_subject(value)
                    or fact.state != value.status
                    or fact.data != _encode_step(value)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not fresh:
            return 0
        fresh_sequences = tuple(value.request_sequence for value in fresh)
        if fresh_sequences != tuple(
            range(durable_count + 1, durable_count + len(fresh) + 1)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if not owner_already_guarded:
            owner_record = await transaction.get_record(self._run_key(run.run_id))
            if owner_record is None or await transaction.guard_record(
                owner_record.key_digest,
                expected_storage_version=owner_record.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        final = await transaction.reserve_sequence(sequence_key, len(fresh))
        if final != fresh_sequences[-1]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stream = self._stream(run.run_id, "interaction")
        owner = self._run_key(run.run_id)
        await _insert_facts(
            transaction,
            tuple(
                StoredFact(
                    stream,
                    value.request_sequence,
                    owner,
                    "model_interaction",
                    _step_subject(value),
                    value.status,
                    _encode_step(value),
                )
                for value in fresh
            ),
        )
        return len(fresh)


class ModelInteractionRuntimeStepStore(RuntimeStepStore):
    """Align staged interaction offsets with the durable request high-water."""

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        await self._ensure_business()
        store = self._staging
        if isinstance(store, ModelInteractionStagingStepStore):
            return await store.list_model_interactions(
                run_id=run_id,
                after_request_sequence=after_request_sequence,
                limit=limit,
            )
        return await super().list_model_interactions(
            run_id=run_id,
            after_request_sequence=after_request_sequence,
            limit=limit,
        )

    async def capture_execution_projection(
        self,
        step_run_id: str,
    ) -> "tuple[CapturedExecutionProjection, _RunProjectionFlight] | None":
        await self._align_execution_interaction_offset(step_run_id)
        return await super().capture_execution_projection(step_run_id)

    async def commit_captured_execution_projection(
        self,
        captured: CapturedExecutionProjection,
        flight: _RunProjectionFlight,
        *,
        execution_id: str,
    ) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if isinstance(archive, ModelInteractionInMemoryStepArchive) and captured.interactions:
            prepared = await archive.prepare_interactions(
                captured.run,
                captured.interactions,
                lambda digest: self._staging.staged_payload(
                    captured.run.run_id,
                    digest,
                ),
                source_messages=(
                    captured.snapshots[-1].messages if captured.snapshots else None
                ),
            )
            captured = replace(captured, interactions=prepared)  # type: ignore[arg-type]
        await super().commit_captured_execution_projection(
            captured,
            flight,
            execution_id=execution_id,
        )

    async def prepare_execution_terminal_seal(
        self,
        *,
        execution_id: str,
        run_ids: Sequence[str],
        binding_digest: str,
    ) -> ExecutionTerminalSealPlan:
        for run_id in dict.fromkeys(run_ids):
            await self._align_execution_interaction_offset(run_id)
        return await super().prepare_execution_terminal_seal(
            execution_id=execution_id,
            run_ids=run_ids,
            binding_digest=binding_digest,
        )

    async def materialize_from_recovery(
        self,
        *,
        target: RuntimeDomain,
        step_run_id: str,
        execution_id: str | None = None,
    ) -> None:
        await super().materialize_from_recovery(
            target=target,
            step_run_id=step_run_id,
            execution_id=execution_id,
        )
        if target is RuntimeDomain.EXECUTION:
            await self._align_execution_interaction_offset(step_run_id)

    async def _align_execution_interaction_offset(self, run_id: str) -> None:
        archive = self._archives.get(RuntimeDomain.EXECUTION)
        if not isinstance(archive, ModelInteractionStateStepArchive):
            return
        head: ExecutionRunSealHead = await archive.execution_history_head_record(run_id)
        async with self._history_lock.hold(run_id):
            offset = self._projection_offsets.setdefault(run_id, _ProjectionOffset())
            offset.interactions = max(offset.interactions, head.interaction_count)


def _validate_interaction_batch(
    run: RunRecord,
    interactions: Sequence[StagedModelInteraction],
) -> tuple[StagedModelInteraction, ...]:
    values = tuple(interactions)
    sequences: set[int] = set()
    for interaction in values:
        if (
            not isinstance(interaction, StagedModelInteraction)
            or interaction.run_id != run.run_id
            or interaction.request_sequence in sequences
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sequences.add(interaction.request_sequence)
    return values


def _required_source_message_count(
    interactions: Sequence[StagedModelInteraction],
) -> int:
    return max(
        (
            max(
                projection.source_message_count,
                max(
                    (
                        item.end
                        for item in projection.items
                        if isinstance(item, StagedContextSpan)
                    ),
                    default=0,
                ),
            )
            for interaction in interactions
            for projection in (
                interaction.request_context,
                *(
                    ()
                    if interaction.response_context is None
                    else (interaction.response_context,)
                ),
            )
        ),
        default=0,
    )


def _validate_interaction_sources(
    interactions: Sequence[StagedModelInteraction],
    source_messages: Sequence[ModelMessage],
) -> None:
    required_count = _required_source_message_count(interactions)
    if required_count > len(source_messages):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    prefix_counts = {
        projection.source_message_count
        for interaction in interactions
        for projection in (
            interaction.request_context,
            *(
                ()
                if interaction.response_context is None
                else (interaction.response_context,)
            ),
        )
        if projection.source_prefix_digest != "0" * 64
    }
    digest = message_prefix_digest(())
    checkpoints = {0: digest} if 0 in prefix_counts else {}
    max_prefix = max(prefix_counts, default=0)
    for index, message in enumerate(source_messages[:max_prefix], 1):
        digest = extend_prefix_digest(digest, message)
        if index in prefix_counts:
            checkpoints[index] = digest
    for interaction in interactions:
        for projection in (
            interaction.request_context,
            *(
                ()
                if interaction.response_context is None
                else (interaction.response_context,)
            ),
        ):
            if projection.source_prefix_digest == "0" * 64:
                continue
            if checkpoints.get(projection.source_message_count) != (
                projection.source_prefix_digest
            ):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "model interaction source prefix digest mismatch",
                )


def _stage_interaction(
    values_by_run: dict[str, list[StagedModelInteraction]],
    interaction: object,
) -> None:
    if not isinstance(interaction, StagedModelInteraction):
        raise TypeError("staged model interaction is invalid")
    values = values_by_run.setdefault(interaction.run_id, [])
    if not values:
        values.append(interaction)
        return
    last_sequence = values[-1].request_sequence
    if interaction.request_sequence == last_sequence + 1:
        values.append(interaction)
        return
    index = interaction.request_sequence - values[0].request_sequence
    if (
        0 <= index < len(values)
        and values[index].request_sequence == interaction.request_sequence
    ):
        if values[index] == interaction:
            return
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _capture_projection(
    store: StagingStepStore,
    run_id: str,
    offset: _ProjectionOffset,
) -> ExecutionProjectionBatch | None:
    store._ensure_open()
    run = store._runs.get(run_id)
    if run is None:
        return None
    events = store._events.get(run_id, ())
    snapshots = store._snapshots.get(run_id, ())
    values = store._interactions.get(run_id, ())
    interactions = tuple(
        value
        for value in values
        if isinstance(value, StagedModelInteraction)
        and value.request_sequence > offset.interactions
    )
    target_interaction = max(
        offset.interactions,
        max(
            (
                value.request_sequence
                for value in values
                if isinstance(value, StagedModelInteraction)
            ),
            default=0,
        ),
    )
    return ExecutionProjectionBatch(
        run,
        tuple(events[offset.events :]),
        tuple(snapshots[offset.snapshots :]),
        offset.events,
        offset.snapshots,
        len(events),
        len(snapshots),
        offset.transcript_messages,
        offset.transcript_messages,
        interactions,
        offset.interactions,
        target_interaction,
    )


def _bounded_interactions(
    values: Sequence[object],
    *,
    after_request_sequence: int | None,
    limit: int | None,
) -> list[object]:
    _validate_page(after_request_sequence, limit)
    selected: list[object] = []
    for value in values:
        if not isinstance(value, (StagedModelInteraction, ModelInteractionRecord)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            after_request_sequence is not None
            and value.request_sequence <= after_request_sequence
        ):
            continue
        selected.append(value)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _validate_page(
    after_request_sequence: int | None,
    limit: int | None,
) -> None:
    if after_request_sequence is not None and (
        isinstance(after_request_sequence, bool)
        or not isinstance(after_request_sequence, int)
        or after_request_sequence < 0
    ):
        raise ValueError("interaction sequence must be non-negative")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("interaction limit must be positive")


__all__ = [
    "ModelInteractionInMemoryStepArchive",
    "ModelInteractionRuntimeStepStore",
    "ModelInteractionStagingStepStore",
    "ModelInteractionStateStepArchive",
]
