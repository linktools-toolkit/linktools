#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint-owned preparation and recoverable archive materialization."""

from collections.abc import Sequence

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._model_interaction import StagedModelInteraction
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence.contracts import ModelInteractionRecord
from linktools.ai.persistence._model_client import resolve_client_requests
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._step_archive import (
    PreparedAgentRunCheckpoint, StagingAgentRunStore, StateStepArchive,
    _materialize_checkpoint,
)
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord, AgentRunStore


class CheckpointPersistence:
    """Prepare recovery data while Runtime retains producer and commit authority."""

    def __init__(
        self, staging: StagingAgentRunStore, archive: AgentRunStore | None,
        trace_archive: AgentRunStore | None,
    ) -> None:
        self._staging = staging
        self.archive = archive
        self._trace_archive = trace_archive

    async def prepare_checkpoint(
        self, archive: StateStepArchive, run: AgentRunRecord,
        checkpoint: AgentRunCheckpoint,
        captured_interactions: Sequence[StagedModelInteraction] | None = None,
    ) -> tuple[PreparedAgentRunCheckpoint, tuple[ModelInteractionRecord, ...], int]:
        checkpoint, batch = await archive.prepare_run_checkpoint(run, checkpoint)
        prepared = batch.checkpoints[0]
        sequence, high_water = await archive.capture_checkpoint_progress(
            agent_run_id=run.agent_run_id, checkpoint=prepared,
        )
        if captured_interactions is None:
            staged = await self.terminal_staged_interactions(
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
            source = self._trace_archive
            if not isinstance(source, StateStepArchive):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            archived = await source.list_model_interactions(
                agent_run_id=run.agent_run_id, after_model_request_seq=high_water,
                limit=staged[0].model_request_seq - high_water - 1,
            )
            if any(not isinstance(value, ModelInteractionRecord) or value.status == "RUNNING" for value in archived):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            projections = await source.resolve_model_interactions(archived)
            resolved = tuple([
                (*projection[:3], await resolve_client_requests(
                    interaction.client_requests, PersistenceDomain.CHECKPOINT, include_bodies=False,
                ), None if interaction.stream_attempts is None else tuple(None for _ in interaction.stream_attempts))
                for interaction, projection in zip(archived, projections, strict=True)
            ])
            missing = await archive.prepare_relocated_interactions(archived, resolved, prepared=prepared.archive)
        interactions = missing + await archive.prepare_interactions(
            run, staged,
            lambda digest: self._staging.staged_payload(run.agent_run_id, digest),
            local_message_count=len(checkpoint.messages),
            local_messages=checkpoint.messages, prepared=prepared.archive,
            **({"baseline": prepared.projection}
               if type(archive).prepare_interactions is StateStepArchive.prepare_interactions else {}),
        )
        return prepared, interactions, sequence


    async def terminal_staged_interactions(
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


    async def materialize_recovery_checkpoint(self, *, agent_run_id: str, require_complete: bool) -> None:
        checkpoint = await self._staging.latest_checkpoint(agent_run_id=agent_run_id, include_interrupted=True)
        run = await self._staging.get_agent_run(agent_run_id=agent_run_id)
        archive = self.archive
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
            interactions = await self.terminal_staged_interactions(
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
                content_write = PreparedArchive()
                prepared = await archive.prepare_interactions(
                    run,
                    tuple(interactions),
                    lambda digest: self._staging.staged_payload(
                        agent_run_id,
                        digest,
                    ),
                    local_message_base=local_base,
                    local_message_count=local_count,
                    local_messages=checkpoint.messages,
                    prepared=content_write,
                    replay=True,
                )
                await archive.sync_projection(
                    run,
                    events=(),
                    checkpoints=(),
                    interactions=prepared,
                    archive=content_write,
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
