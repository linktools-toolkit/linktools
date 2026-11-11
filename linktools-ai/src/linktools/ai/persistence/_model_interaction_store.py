#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-interaction archives and staging persistence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from ..errors import AIError, ErrorCode
from .contracts import (
    ExecutionRecord,
    ExecutionInsertionReceipt,
    ModelInteractionRecord,
)
from ._history import PreparedTranscriptObservation
from ._step_contracts import AgentRunRecord, StepEvent
from ._step_archive import (
    PreparedAgentRunCheckpoint,
    StateStepArchive,
    _decode_step,
    _encode_step,
    _validate_interaction_page,
)
from .state import (
    RecordQuery,
    RecordReplacement,
    StateStore,
    StateTransaction,
    StoredRecord,
    record_key_digest,
    require_no_run_history_lock,
)


class ModelInteractionStateStepArchive(StateStepArchive):
    """Durable archive with one CAS-protected record per admitted model request."""

    async def capture_model_interactions(
        self,
        interactions: Sequence[object],
        *,
        include_client_requests: bool = False,
        include_content: bool = True,
    ) -> Callable[[], Awaitable[list[object]]]:
        values = tuple(interactions)
        if not values:
            return await super().capture_model_interactions(())
        if any(not isinstance(value, ModelInteractionRecord) for value in values):
            raise TypeError("model interaction is invalid")
        records = tuple(
            value for value in values if isinstance(value, ModelInteractionRecord)
        )
        if any(record.agent_run_id != records[0].agent_run_id for record in records):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await super().capture_model_interactions(
            records, include_client_requests=include_client_requests, include_content=include_content,
        )

    def _interaction_key(self, agent_run_id: str, model_request_seq: int) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            self.runtime_domain.value,
            "model_interaction",
            [agent_run_id, model_request_seq],
        )

    async def checkpoint_publication_record_keys(
        self, run: AgentRunRecord, checkpoint: PreparedAgentRunCheckpoint,
        *, interactions: Sequence[ModelInteractionRecord] = (),
    ) -> Mapping[StateStore, tuple[bytes, ...]]:
        keys = dict(await super().checkpoint_publication_record_keys(run, checkpoint))
        if all(isinstance(value, ModelInteractionRecord) and value.agent_run_id == run.agent_run_id
               for value in interactions):
            keys[self.state_store] = tuple(dict.fromkeys((
                *keys.get(self.state_store, ()),
                *(self._interaction_key(value.agent_run_id, value.model_request_seq) for value in interactions),
            )))
        return keys

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        _, interactions = await self._read_model_interactions(
            agent_run_id=agent_run_id, after_model_request_seq=after_model_request_seq,
            limit=limit, include_run=False,
        )
        return interactions

    async def read_model_interactions_with_run(
        self, *, agent_run_id: str, after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> tuple[AgentRunRecord | None, list[object]]:
        """Read the owning run and a bounded interaction page in one state scope."""
        return await self._read_model_interactions(
            agent_run_id=agent_run_id, after_model_request_seq=after_model_request_seq,
            limit=limit, include_run=True,
        )

    async def _read_model_interactions(
        self, *, agent_run_id: str, after_model_request_seq: int | None,
        limit: int | None, include_run: bool,
    ) -> tuple[AgentRunRecord | None, list[object]]:
        self._ensure_open()
        require_no_run_history_lock("StateStepArchive.list_model_interactions")
        _validate_interaction_page(after_model_request_seq, limit)
        sequence_key = self._stream(agent_run_id, "interaction")
        start = 1 if after_model_request_seq is None else after_model_request_seq + 1

        async def read(transaction: StateTransaction) -> tuple[AgentRunRecord | None, list[object]]:
            high_water = (await transaction.get_sequences((sequence_key,))).get(sequence_key, 0)
            end = high_water + 1 if limit is None else min(high_water + 1, start + limit)
            keys = tuple(self._interaction_key(agent_run_id, seq) for seq in range(start, end))
            run_key = self._agent_run_key(agent_run_id) if include_run else None
            if not keys and run_key is None:
                return None, []
            records = (await transaction.get_records(keys) if run_key is None
                       else await self.read_run_records(transaction, (*keys, run_key)))
            stored_run = records.get(run_key) if run_key is not None else None
            run = None if stored_run is None else _decode_step(stored_run.data)
            if run is not None and not isinstance(run, AgentRunRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result: list[object] = []
            for key in keys:
                record = records.get(key)
                if record is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                result.append(self._decode_interaction_record(record))
            return run, result

        return await self._store.read(read)

    def _stored_interaction(self, value: ModelInteractionRecord) -> StoredRecord:
        value.validate_storage_domain(self.runtime_domain)
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
        value.validate_storage_domain(self.runtime_domain)
        if (
            record.key_digest != self._interaction_key(value.agent_run_id, value.model_request_seq)
            or record.scope_key is not None
            or record.parent_key != self._agent_run_key(value.agent_run_id)
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
            await self.guard_execution_producer(
                transaction, execution_id=execution_id, producer_generation=producer_generation,
            )
            owner_key = self._agent_run_key(agent_run_id)
            owner = (await self.read_run_records(transaction, (owner_key,))).get(owner_key)
            if owner is None:
                return
            run = _decode_step(owner.data)
            if not isinstance(run, AgentRunRecord) or run.agent_run_id != agent_run_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            records = await transaction.list_records(
                RecordQuery(
                    parent_key=owner.key_digest,
                    kind="model_interaction",
                    states=frozenset({"RUNNING"}),
                )
            )
            interactions = tuple(
                replace(self._decode_interaction_record(record), status="INTERRUPTED")
                for record in records
            )
            await self._sync_interactions_in_transaction(
                transaction,
                run,
                interactions,
                owner_already_guarded=False,
            )

        await self._mutate(mutate)

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
        execution_guard: tuple[ExecutionRecord, StoredRecord] | None = None,
        inserted_terminal: ExecutionInsertionReceipt | None = None,
    ) -> bool:
        guard = await self._execution_history_guard_in_transaction(
            transaction,
            execution_id,
            execution_guard,
            producer_generation=producer_generation,
            inserted_terminal=inserted_terminal,
        )
        # Invalid inputs retain the original validation/error ordering below.
        prefetch_keys = (
            tuple(self._interaction_key(value.agent_run_id, value.model_request_seq) for value in interactions)
            if (guard is not None or producer_generation is not None) and all(
                isinstance(value, ModelInteractionRecord) and value.agent_run_id == run.agent_run_id
                for value in interactions
            )
            else ()
        )
        created = await super()._sync_projection_in_transaction(
            transaction,
            run,
            events=events,
            checkpoints=checkpoints,
            interactions=(),
            observation=observation,
            producer_generation=producer_generation,
            inserted_terminal=inserted_terminal,
            execution_id=execution_id,
            execution_guard=guard,
            prefetch_keys=prefetch_keys,
        )
        inserted = await self._sync_interactions_in_transaction(
            transaction,
            run,
            interactions,
            owner_already_guarded=bool(events or checkpoints),
        )
        return created

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

        sequence_key = self._stream(run.agent_run_id, "interaction")
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
                            storage_revision=previous.storage_revision + 1,
                        ),
                        previous.storage_revision,
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
                expected_storage_revision=owner_record.storage_revision,
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


def _interaction_sort_key(model_request_seq: int) -> bytes:
    return f"m:{model_request_seq:020d}".encode("ascii")


__all__ = [
    "ModelInteractionStateStepArchive",
]
