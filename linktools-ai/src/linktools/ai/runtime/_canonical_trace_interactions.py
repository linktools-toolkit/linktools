"""Prepare exact SDK request spans for the canonical Trace owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..persistence.runs import ModelInteractionStateStepArchive, PreparedArchive

if TYPE_CHECKING:
    from ..persistence.runs import AgentRunRecord, StagingAgentRunStore, StagedModelInteraction, StepEvent
    from pydantic_ai.messages import ModelMessage


@dataclass(frozen=True, slots=True)
class _CapturedTurnInteractions:
    """An immutable, producer-local batch detached before recorder teardown."""

    run: AgentRunRecord
    interactions: tuple[StagedModelInteraction, ...]
    events: tuple[StepEvent, ...]
    local_messages: tuple[ModelMessage, ...]
    payloads: dict[str, bytes]

    @classmethod
    def capture(
        cls, staging: StagingAgentRunStore, turn_id: str, session_id: str, *,
        after: int = 0, after_events: int = 0, settled_only: bool = False,
    ) -> _CapturedTurnInteractions | None:
        interactions = tuple(value for value in staging.capture_interactions_local(turn_id)
                             if value.model_request_seq > after)
        if settled_only:
            from itertools import takewhile
            interactions = tuple(takewhile(lambda item: item.status != "RUNNING", interactions))
        events = tuple(event for event in staging.list_events_local(turn_id)
                       if event.event_type.startswith(("TOOL_CALL_", "MODEL_REQUEST_")))[after_events:]
        if not interactions and not events:
            return None
        run = staging.get_agent_run_local(turn_id)
        if run is None or run.agent_run_id != turn_id or run.agent_conversation_id != session_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        transcript = staging.staged_transcript(turn_id)
        dependencies = {digest for item in interactions for digest, _ in item.payload_dependencies()}
        return cls(
            run, interactions, events, () if transcript is None else tuple(transcript.messages),
            {digest: staging.staged_payload(turn_id, digest) for digest in dependencies},
        )

    async def prepare_append(self, archive: ModelInteractionStateStepArchive):
        """Prepare this observation batch for the independent Trace transaction."""
        prepared = PreparedArchive()
        content = await archive.transcript_repository.content_archive(create=True, best_effort=True)
        if content.store.storage_group is not archive.state_store.storage_group:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        values = await archive.prepare_interactions(
            self.run, self.interactions, self.payloads.__getitem__,
            local_message_count=len(self.local_messages), local_messages=self.local_messages, prepared=prepared,
        )

        async def append(transaction, run: AgentRunRecord) -> None:
            trace = transaction.transaction(archive.state_store)
            await content.publish(transaction.transaction(content.store), prepared)
            await archive.append_canonical_turn_interactions_in_transaction(trace, run, interactions=values)

        return content.store, append
