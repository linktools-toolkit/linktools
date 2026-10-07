#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-interaction archives and staging persistence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from ...errors import AIError, ErrorCode
from ...storage import StoredPayload
from .._model_interaction import (
    StagedContextProjection,
    StagedContextSpan,
    StagedModelInteraction,
    context_projection_to_durable,
)
from ._contracts import (
    ContextProjection,
    ExecutionHistoryHeadRecord,
    InlineContextBlock,
    ModelInteractionRecord,
    RuntimePayloadRef,
    TranscriptSpanRef,
)
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from ._step_archive import (
    ExecutionProjectionBatch,
    InMemoryStepArchive,
    PreparedAgentRunCheckpoint,
    StagingAgentRunStore,
    StateStepArchive,
    _ProjectionOffset,
    _decode_step,
    _encode_step,
    _insert_facts,
    _step_subject,
)
from ._store import FactQuery, StateTransaction, StoredFact, StoredRecord


class ModelInteractionStagingAgentRunStore(StagingAgentRunStore):
    """Staging store with model-request-seq idempotency and durable high-water capture."""

    def capture_projection_local(
        self,
        agent_run_id: str,
        offset: _ProjectionOffset,
    ) -> ExecutionProjectionBatch | None:
        # Base staging owns runs/events/checkpoints/interactions. Ask it for a
        # complete local interaction checkpoint, then translate the durable
        # model-request-seq high-water without reaching into its other state.
        base = super().capture_projection_local(
            agent_run_id,
            replace(offset, interactions=0),
        )
        if base is None:
            return None
        return _with_interaction_high_water(base, offset.interactions)


class ModelInteractionInMemoryStepArchive(InMemoryStepArchive):
    """Volatile archive using the same resolved interaction contract as durable stores."""

    async def prepare_interactions(
        self,
        run: AgentRunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        *,
        local_message_base: int = 0,
        local_message_count: int = 0,
    ) -> tuple[ModelInteractionRecord, ...]:
        values = _validate_interaction_batch(run, interactions)
        if local_message_base < 0 or local_message_count < 0:
            raise ValueError("local interaction transcript range is invalid")

        def prepare_context(staged: StagedContextProjection) -> ContextProjection:
            for item in staged.items:
                if isinstance(item, StagedContextSpan) and item.end > local_message_count:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if isinstance(item, TranscriptSpanRef) and (
                    item.source_domain is not self.runtime_domain
                    or item.owner_id != run.agent_run_id
                ):
                    raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            return context_projection_to_durable(
                staged,
                owner_id=run.agent_run_id,
                source_domain=self.runtime_domain,
                payload=payload,
                local_message_base=local_message_base,
            )

        return tuple(
            ModelInteractionRecord(
                staged.agent_run_id,
                staged.step_index,
                staged.model_request_seq,
                staged.purpose,
                staged.output_retry_index,
                staged.model,
                prepare_context(staged.request_context),
                RuntimePayloadRef(
                    StoredPayload.inline_bytes(payload(staged.request_envelope_digest)),
                    self.runtime_domain,
                ),
                None
                if staged.response_context is None
                else prepare_context(staged.response_context),
                staged.status,
                staged.error_code,
                staged.duration_ns,
                staged.usage,
                staged.started_at,
                staged.finished_at,
                staged.attachments,
            )
            for staged in values
        )

    async def sync_projection(
        self,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[AgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        execution_id: str | None = None,
    ) -> None:
        current_values = await super().list_model_interactions(agent_run_id=run.agent_run_id)
        current: dict[int, ModelInteractionRecord] = {}
        for value in current_values:
            if not isinstance(value, ModelInteractionRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current[value.model_request_seq] = value
        fresh: list[ModelInteractionRecord] = []
        for interaction in interactions:
            if not isinstance(interaction, ModelInteractionRecord):
                raise TypeError("model interaction is invalid")
            previous = current.get(interaction.model_request_seq)
            if previous is None:
                fresh.append(interaction)
                current[interaction.model_request_seq] = interaction
            elif previous != interaction:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await super().sync_projection(
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=fresh,
            execution_id=execution_id,
        )

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        values = await super().list_model_interactions(
            agent_run_id=agent_run_id,
            after_model_request_seq=after_model_request_seq,
            limit=limit,
        )
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return values


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
        if any(record.agent_run_id != records[0].agent_run_id for record in records):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await super().resolve_model_interactions(records)

    async def prepare_interactions(
        self,
        run: AgentRunRecord,
        interactions: Sequence[StagedModelInteraction],
        payload: Callable[[str], bytes],
        *,
        local_message_base: int = 0,
        local_message_count: int = 0,
    ) -> tuple[ModelInteractionRecord, ...]:
        return await super().prepare_interactions(
            run,
            interactions,
            payload,
            local_message_base=local_message_base,
            local_message_count=local_message_count,
        )

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        # FactQuery is the physical pagination boundary and owns validation of
        # after_sequence/limit. Do not duplicate that contract here.
        values = await self._store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    self._stream(agent_run_id, "interaction"),
                    after_sequence=after_model_request_seq,
                    limit=limit,
                )
            )
        )
        return [self._decode_interaction_fact(fact) for fact in values]

    def _decode_interaction_fact(self, fact: StoredFact) -> ModelInteractionRecord:
        value = _decode_step(fact.data)
        if (
            not isinstance(value, ModelInteractionRecord)
            or fact.sequence != value.model_request_seq
            or fact.subject_digest != _step_subject(value)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
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
        owner = self._agent_run_key(run.agent_run_id)
        run_existed = await transaction.get_record(owner) is not None
        await super()._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=(),
            execution_id=execution_id,
            history_head_guard=guard,
        )
        inserted = await self._sync_interactions_in_transaction(
            transaction,
            run,
            interactions,
            owner_already_guarded=bool(events or checkpoints),
        )
        if guard is not None and not supplied_guard and (
            not run_existed or events or checkpoints or inserted
        ):
            await self._advance_execution_history_head_in_transaction(
                transaction,
                guard,
            )

    async def _sync_interactions_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        interactions: Sequence[ModelInteractionRecord],
        *,
        owner_already_guarded: bool,
    ) -> int:
        values = tuple(interactions)
        if not values:
            return 0
        sequences = tuple(value.model_request_seq for value in values)
        if (
            any(value.agent_run_id != run.agent_run_id for value in values)
            or sequences != tuple(sorted(sequences))
            or len(set(sequences)) != len(sequences)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        sequence_key = self._sequence(run.agent_run_id, "interaction")
        durable_count = (await transaction.get_sequences((sequence_key,))).get(
            sequence_key,
            0,
        )
        replay = tuple(
            value for value in values if value.model_request_seq <= durable_count
        )
        fresh = tuple(
            value for value in values if value.model_request_seq > durable_count
        )
        if replay:
            replay_sequences = tuple(value.model_request_seq for value in replay)
            if replay_sequences != tuple(
                range(
                    replay_sequences[0],
                    replay_sequences[0] + len(replay_sequences),
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            stored = await transaction.list_facts(
                FactQuery(
                    self._stream(run.agent_run_id, "interaction"),
                    after_sequence=replay_sequences[0] - 1,
                    limit=len(replay),
                )
            )
            if len(stored) != len(replay):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for fact, value in zip(stored, replay, strict=True):
                if (
                    fact.sequence != value.model_request_seq
                    or fact.subject_digest != _step_subject(value)
                    or fact.state != value.status
                    or fact.data != _encode_step(value)
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not fresh:
            return 0
        fresh_sequences = tuple(value.model_request_seq for value in fresh)
        if fresh_sequences != tuple(
            range(durable_count + 1, durable_count + len(fresh) + 1)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if not owner_already_guarded:
            owner_record = await transaction.get_record(self._agent_run_key(run.agent_run_id))
            if owner_record is None or await transaction.guard_record(
                owner_record.key_digest,
                expected_storage_version=owner_record.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        final = await transaction.reserve_sequence(sequence_key, len(fresh))
        if final != fresh_sequences[-1]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stream = self._stream(run.agent_run_id, "interaction")
        owner = self._agent_run_key(run.agent_run_id)
        await _insert_facts(
            transaction,
            tuple(
                StoredFact(
                    stream,
                    value.model_request_seq,
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


def _validate_interaction_batch(
    run: AgentRunRecord,
    interactions: Sequence[StagedModelInteraction],
) -> tuple[StagedModelInteraction, ...]:
    values = tuple(interactions)
    sequences: set[int] = set()
    for interaction in values:
        if (
            not isinstance(interaction, StagedModelInteraction)
            or interaction.agent_run_id != run.agent_run_id
            or interaction.model_request_seq in sequences
            or interaction.status == "RUNNING"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sequences.add(interaction.model_request_seq)
    return values



def _with_interaction_high_water(
    batch: ExecutionProjectionBatch,
    durable_high_water: int,
) -> ExecutionProjectionBatch:
    interactions: list[StagedModelInteraction] = []
    expected = durable_high_water + 1
    running_seen = False
    for value in batch.interactions:
        if value.model_request_seq <= durable_high_water:
            continue
        if value.model_request_seq != expected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value.status == "RUNNING":
            running_seen = True
            continue
        if running_seen:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        interactions.append(value)
        expected += 1
    target = durable_high_water + len(interactions)
    return replace(
        batch,
        interactions=tuple(interactions),
        base_interaction_offset=durable_high_water,
        target_interaction_offset=target,
    )


__all__ = [
    "ModelInteractionInMemoryStepArchive",
    "ModelInteractionStagingAgentRunStore",
    "ModelInteractionStateStepArchive",
]
