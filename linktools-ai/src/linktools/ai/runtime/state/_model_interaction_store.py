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
from ._history import PreparedTranscriptObservation
from ._plan import RuntimeDomain
from ._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent
from ._step_archive import (
    InMemoryStepArchive,
    PreparedAgentRunCheckpoint,
    StateStepArchive,
    _decode_step,
    _encode_step,
    _validate_interaction_page,
)
from ._store import (
    RecordQuery,
    RecordReplacement,
    StateTransaction,
    StoredRecord,
    record_key_digest,
    require_no_run_history_lock,
)


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
        values = tuple(
            replace(value, request_context=None, request_envelope_digest=None)
            if self.runtime_domain is RuntimeDomain.EXECUTION and value.status == "RUNNING"
            else value
            for value in _validate_interaction_batch(run, interactions)
        )
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
                None
                if staged.request_context is None
                else prepare_context(staged.request_context),
                None
                if staged.request_envelope_digest is None
                else RuntimePayloadRef(
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
        fresh: list[ModelInteractionRecord] = []
        seen: set[int] = set()
        for interaction in interactions:
            if not isinstance(interaction, ModelInteractionRecord):
                raise TypeError("model interaction is invalid")
            if interaction.agent_run_id != run.agent_run_id or interaction.model_request_seq in seen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(interaction.model_request_seq)
            existing = await super().list_model_interactions(
                agent_run_id=run.agent_run_id,
                after_model_request_seq=interaction.model_request_seq - 1,
                limit=1,
            )
            previous = existing[0] if existing else None
            if previous is None or previous.model_request_seq != interaction.model_request_seq:
                fresh.append(interaction)
            else:
                if not isinstance(previous, ModelInteractionRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                previous.validate_successor(interaction)
                if previous != interaction:
                    fresh.append(interaction)
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
    """Durable archive with one CAS-protected record per admitted model request."""

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

    def _interaction_key(self, agent_run_id: str, model_request_seq: int) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self.runtime_domain.value,
            "model_interaction",
            [agent_run_id, model_request_seq],
        )

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.list_model_interactions")
        _validate_interaction_page(after_model_request_seq, limit)
        sequence_key = self._sequence(agent_run_id, "interaction")
        start = 1 if after_model_request_seq is None else after_model_request_seq + 1

        async def read(transaction: StateTransaction) -> list[object]:
            high_water = (await transaction.get_sequences((sequence_key,))).get(sequence_key, 0)
            end = high_water + 1 if limit is None else min(high_water + 1, start + limit)
            keys = tuple(self._interaction_key(agent_run_id, seq) for seq in range(start, end))
            if not keys:
                return []
            records = await transaction.get_records(keys)
            if len(records) != len(keys):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return [self._decode_interaction_record(records[key]) for key in keys]

        return await self._store.read(read)

    def _stored_interaction(self, value: ModelInteractionRecord) -> StoredRecord:
        return StoredRecord(
            self._interaction_key(value.agent_run_id, value.model_request_seq),
            None,
            self._agent_run_key(value.agent_run_id),
            "model_interaction",
            _interaction_sort_key(value.model_request_seq),
            value.status,
            0,
            None,
            0,
            None,
            _encode_step(value),
        )

    def _decode_interaction_record(self, record: StoredRecord) -> ModelInteractionRecord:
        value = _decode_step(record.data)
        if not isinstance(value, ModelInteractionRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            record.key_digest != self._interaction_key(value.agent_run_id, value.model_request_seq)
            or record.scope_digest is not None
            or record.parent_digest != self._agent_run_key(value.agent_run_id)
            or record.kind != "model_interaction"
            or record.sort_key != _interaction_sort_key(value.model_request_seq)
            or record.state != value.status
            or record.lease_owner is not None
            or record.lease_fence != 0
            or record.lease_expires_at is not None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def interrupt_model_interactions(
        self,
        *,
        agent_run_id: str,
        execution_id: str,
        producer_generation: int,
    ) -> None:
        """Close requests left in flight by a previous execution producer."""
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.interrupt_model_interactions")

        async def mutate(transaction: StateTransaction) -> None:
            guard = await self._execution_history_guard_in_transaction(
                transaction,
                execution_id,
                None,
                producer_generation=producer_generation,
            )
            if guard is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            owner = await transaction.get_record(self._agent_run_key(agent_run_id))
            if owner is None:
                return
            run = _decode_step(owner.data)
            if not isinstance(run, AgentRunRecord) or run.agent_run_id != agent_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            records = await transaction.list_records(
                RecordQuery(
                    parent_digest=owner.key_digest,
                    kind="model_interaction",
                    states=frozenset({"RUNNING"}),
                )
            )
            interactions = tuple(
                replace(self._decode_interaction_record(record), status="INTERRUPTED")
                for record in records
            )
            changed = await self._sync_interactions_in_transaction(
                transaction,
                run,
                interactions,
                owner_already_guarded=False,
            )
            if changed:
                await self._advance_execution_history_head_in_transaction(transaction, guard)

        await self._store.mutate(mutate)

    async def _sync_projection_in_transaction(
        self,
        transaction: StateTransaction,
        run: AgentRunRecord,
        *,
        events: Sequence[StepEvent],
        checkpoints: Sequence[PreparedAgentRunCheckpoint],
        interactions: Sequence[ModelInteractionRecord] = (),
        observation: PreparedTranscriptObservation | None = None,
        producer_generation: int | None = None,
        execution_id: str | None = None,
        history_head_guard: tuple[ExecutionHistoryHeadRecord, StoredRecord] | None = None,
    ) -> None:
        supplied_guard = history_head_guard is not None
        guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            history_head_guard,
            producer_generation=producer_generation,
        )
        owner = self._agent_run_key(run.agent_run_id)
        run_existed = await transaction.get_record(owner) is not None
        await super()._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=(),
            observation=observation,
            producer_generation=producer_generation,
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
            not run_existed or events or checkpoints or inserted or observation is not None
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
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise TypeError("model interaction is invalid")
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
        keys = tuple(
            self._interaction_key(value.agent_run_id, value.model_request_seq)
            for value in values
        )
        stored = await transaction.get_records(keys)
        fresh: list[StoredRecord] = []
        replacements: list[RecordReplacement] = []
        for key, value in zip(keys, values, strict=True):
            previous = stored.get(key)
            if value.model_request_seq > durable_count:
                if previous is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fresh.append(self._stored_interaction(value))
                continue
            if previous is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_value = self._decode_interaction_record(previous)
            previous_value.validate_successor(value)
            if previous_value != value:
                replacements.append(
                    RecordReplacement(
                        replace(
                            previous,
                            state=value.status,
                            data=_encode_step(value),
                            storage_version=previous.storage_version + 1,
                        ),
                        previous.storage_version,
                    )
                )
        fresh_sequences = tuple(
            value.model_request_seq for value in values
            if value.model_request_seq > durable_count
        )
        if fresh_sequences != tuple(
            range(durable_count + 1, durable_count + len(fresh) + 1)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not fresh and not replacements:
            return 0
        if not owner_already_guarded:
            owner_record = await transaction.get_record(self._agent_run_key(run.agent_run_id))
            if owner_record is None or await transaction.guard_record(
                owner_record.key_digest,
                expected_storage_version=owner_record.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        if fresh:
            final = await transaction.reserve_sequence(sequence_key, len(fresh))
            if final != fresh_sequences[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await transaction.insert_records(tuple(fresh))
        if replacements:
            await transaction.replace_records(tuple(replacements))
        return len(fresh) + len(replacements)


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
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sequences.add(interaction.model_request_seq)
    return values



def _interaction_sort_key(model_request_seq: int) -> str:
    return f"m:{model_request_seq:020d}"


__all__ = [
    "ModelInteractionInMemoryStepArchive",
    "ModelInteractionStateStepArchive",
]
