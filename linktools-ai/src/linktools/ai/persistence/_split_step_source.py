#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subordinate SQL fences for a canonical turn's separately stored Step run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..core import JsonValue, validate_resource_id
from ..errors import AIError, ErrorCode
from ..steps import decode_native_replay_key, encode_native_replay_key, native_store_locator
from ._message_root import NativeMessageRoot
from ._native_content_index import stage_owner

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import Table


@dataclass(frozen=True, slots=True)
class SplitStepWaiting:
    """One waiting leaf on the enclosing source generation's exact checkpoint."""

    kind: Literal["approval", "external"]
    operation_id: str
    receipt_id: str
    original_generation: int

    def to_json(self) -> dict:
        if self.kind not in ("approval", "external"):
            raise ValueError("invalid split waiting leaf kind")
        validate_resource_id(self.operation_id)
        validate_resource_id(self.receipt_id)
        if type(self.original_generation) is not int or self.original_generation < 1:
            raise ValueError("invalid split waiting generation")
        return {
            "kind": self.kind, "operation_id": self.operation_id, "receipt_id": self.receipt_id,
            "original_generation": self.original_generation,
        }

    @classmethod
    def from_json(cls, value: dict) -> SplitStepWaiting:
        if value.keys() != {"kind", "operation_id", "receipt_id", "original_generation"}:
            raise ValueError("invalid split waiting leaf")
        return cls(value["kind"], value["operation_id"], value["receipt_id"], value["original_generation"])


@dataclass(frozen=True, slots=True)
class SplitStepPin:
    receipt_id: str
    checkpoint_id: str | None = None
    terminal_state: str | None = None
    transcript: NativeMessageRoot | None = None
    result: JsonValue = None
    binding_refs: tuple[tuple[int, int | None], ...] = ()
    output_ref: tuple[int, int] | None = None
    waiting: SplitStepWaiting | None = None
    terminal_text_bytes: int = 0

    @classmethod
    def for_terminal(
        cls, receipt_id: str, *, terminal_message: object, terminal_state: str,
        transcript: NativeMessageRoot | None, result: JsonValue,
        binding_refs: tuple[tuple[int, int | None], ...], output_ref: tuple[int, int] | None,
    ) -> SplitStepPin:
        from ._session_transcript import terminal_text_bytes

        return cls(receipt_id, terminal_state=terminal_state, transcript=transcript, result=result,
                   binding_refs=binding_refs, output_ref=output_ref,
                   terminal_text_bytes=terminal_text_bytes(terminal_message))

    def to_json(self) -> dict:
        validate_resource_id(self.receipt_id)
        if len(self.receipt_id.encode("utf-8")) > 64:
            raise ValueError("split source receipt is too wide")
        if self.checkpoint_id is not None:
            encode_native_replay_key(self.checkpoint_id)
        if self.terminal_state not in (None, "succeeded", "failed", "cancelled"):
            raise ValueError("invalid split terminal intent")
        if self.waiting is not None and (
            self.terminal_state is not None or self.checkpoint_id is None
        ):
            raise ValueError("split waiting intent requires an exact nonterminal checkpoint")
        if type(self.terminal_text_bytes) is not int or self.terminal_text_bytes < 0 or (
                self.terminal_state != "succeeded" and self.terminal_text_bytes != 0):
            raise ValueError("invalid split terminal text charge")
        if self.terminal_state == "succeeded":
            if self.transcript is None or not self.transcript.count or self.result is not None:
                raise ValueError("successful split receipt requires its native transcript")
            for index, part in self.binding_refs:
                if (type(index) is not int or not 0 <= index < self.transcript.count
                        or (part is not None and (type(part) is not int or part < 0))):
                    raise ValueError("invalid split transcript binding reference")
            if self.output_ref is not None:
                index, part = self.output_ref
                if (type(index) is not int or not 0 <= index < self.transcript.count
                        or type(part) is not int or part < 0):
                    raise ValueError("invalid split transcript output reference")
        elif self.transcript is not None or self.binding_refs or self.output_ref is not None:
            raise ValueError("non-success split receipt cannot carry transcript metadata")
        elif self.terminal_state is None and self.result is not None:
            raise ValueError("takeover receipt cannot carry a terminal result")
        return {
            "result": self.result,
            "binding_refs": [list(ref) for ref in self.binding_refs],
            "output_ref": None if self.output_ref is None else list(self.output_ref),
            "checkpoint_id": self.checkpoint_id,
            "terminal_state": self.terminal_state,
            **({"terminal_text_bytes": self.terminal_text_bytes} if self.terminal_state == "succeeded" else {}),
            "transcript": None if self.transcript is None else self.transcript.to_json(),
            **({} if self.waiting is None else {"waiting": self.waiting.to_json()}),
        }


    @classmethod
    def from_json(cls, receipt_id: str, value: dict) -> SplitStepPin:
        fields = {"checkpoint_id", "terminal_state", "transcript", "result", "binding_refs", "output_ref"}
        if value.get("terminal_state") == "succeeded":
            fields.add("terminal_text_bytes")
        if value.keys() not in (fields, fields | {"waiting"}):
            raise ValueError("invalid split source pin")
        transcript = value["transcript"]
        output_ref = value["output_ref"]
        pin = cls(
            receipt_id, checkpoint_id=value["checkpoint_id"], terminal_state=value["terminal_state"],
            transcript=None if transcript is None else NativeMessageRoot.from_json(transcript),
            result=value["result"], terminal_text_bytes=value.get("terminal_text_bytes", 0), binding_refs=tuple(tuple(ref) for ref in value["binding_refs"]),
            output_ref=None if output_ref is None else tuple(output_ref),
            waiting=SplitStepWaiting.from_json(value["waiting"]) if "waiting" in value else None,
        )
        pin.to_json()
        return pin

@dataclass(frozen=True, slots=True)
class SplitStepSourceObservation:
    turn_id: str
    session_id: str
    generation: int
    phase: str
    pin: SplitStepPin


class SqlSplitStepSource:
    """Operate inside the caller's source transaction, never against Conversation."""

    def __init__(self, runs: Table, snapshots: Table, *, store_key: bytes) -> None:
        self._runs, self._snapshots = runs, snapshots
        self.store_key = store_key

    def _where(self, table: Table, *, turn_id: str, session_id: str):
        return (
            table.c.store_locator == native_store_locator(self.store_key),
            table.c.store_key == self.store_key,
            table.c.run_id == turn_id,
            table.c.conversation_id == session_id,
        )

    async def adopt(
        self, session: AsyncSession, *, turn_id: str, session_id: str, generation: int,
        predecessor: SplitStepPin | None = None,
    ) -> bool:
        """Adopt an already committed target generation after exact source sealing.

        A caller must first observe Conversation's exact new generation; never
        invoke this for a speculative claim or an unknown target CAS.
        """
        if generation < 1 or (predecessor is None) != (generation == 1):
            raise ValueError("split source adoption requires an exact predecessor")
        if predecessor is not None and predecessor.waiting is not None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        columns = self._runs.c
        previous = ((columns.turn_source_generation.is_(None),) if predecessor is None else (
            columns.turn_source_generation == generation - 1,
            columns.turn_source_phase == "sealed",
            columns.turn_source_receipt == predecessor.receipt_id,
        ))
        return await self._adopt(session, turn_id=turn_id, session_id=session_id, generation=generation,
                                 previous=previous)

    async def adopt_ready(self, session: AsyncSession, source: SplitStepSourceObservation) -> bool:
        """Open only after the target grant of this exact settled waiting receipt."""
        if source.phase != "ready" or source.pin.waiting is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase == "ready",
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_generation=source.generation + 1, turn_source_phase="resuming"))
        return changed.rowcount == 1

    async def freeze_resuming(
        self, session: AsyncSession, source: SplitStepSourceObservation, *, expired_generation: int,
    ) -> bool:
        """Fence even the previous owner's first checkpoint before target takeover."""
        if (source.pin.waiting is None or source.phase not in ("ready", "resuming", "resume_hold")
                or source.generation not in (expired_generation - 1, expired_generation)):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return await self._move_waiting_pin(session, source, generation=expired_generation, phase="resume_hold")

    async def adopt_frozen_resume(self, session: AsyncSession, source: SplitStepSourceObservation) -> bool:
        """Reopen only the retained source following a proven target grant."""
        if source.phase != "resume_hold" or source.pin.waiting is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        return await self._move_waiting_pin(session, source, generation=source.generation + 1, phase="resuming")

    async def _move_waiting_pin(
        self, session: AsyncSession, source: SplitStepSourceObservation, *, generation: int, phase: str,
    ) -> bool:
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase == source.phase,
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_generation=generation, turn_source_phase=phase))
        return changed.rowcount == 1

    async def guard_resuming_approval(
        self, session: AsyncSession, source: SplitStepSourceObservation,
    ) -> None:
        waiting = source.pin.waiting
        if source.phase != "resuming" or waiting is None or waiting.kind != "approval":
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        # Acquire the same write fence as source freeze/consume, without
        # advancing the phase or replacing the retained checkpoint.
        if not await self._move_waiting_pin(
            session, source, generation=source.generation, phase="resuming",
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if await self.observe_pin(
            session, turn_id=source.turn_id, session_id=source.session_id, generation=source.generation,
        ) != source:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def consume_resumed_checkpoint(self, session: AsyncSession, source: SplitStepSourceObservation) -> bool:
        """Remove the retained pin only inside its first new checkpoint transaction."""
        if source.phase != "resuming" or source.pin.waiting is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase == "resuming",
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_phase="open", turn_source_receipt=None, turn_source_pin=None))
        return changed.rowcount == 1

    async def _adopt(
        self, session: AsyncSession, *, turn_id: str, session_id: str, generation: int, previous: tuple,
    ) -> bool:
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id), *previous,
        ).values(
            turn_source_generation=generation, turn_source_phase="open",
            turn_source_receipt=None, turn_source_pin=None,
        ))
        return changed.rowcount == 1

    async def cancel_unregistered(
        self, session: AsyncSession, *, turn_id: str, session_id: str,
        admitted_at: str, pin: SplitStepPin,
    ) -> SplitStepSourceObservation:
        """Claim the exact unique identity before its first writer can."""
        if pin.terminal_state != "cancelled" or pin.checkpoint_id is not None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        await session.execute(self._runs.insert().values(
            store_locator=native_store_locator(self.store_key), store_key=self.store_key,
            run_id=turn_id, conversation_id=session_id, metadata="{}", started_at=admitted_at,
            turn_source_generation=1, turn_source_phase="sealed",
            turn_source_receipt=pin.receipt_id, turn_source_pin=pin.to_json(),
        ))
        # Deliberately metadata-only: no SDK registration identity, managed
        # start, events or result body. A late writer cannot win this key.
        return SplitStepSourceObservation(turn_id, session_id, 1, "sealed", pin)

    async def has_registration(self, session: AsyncSession, *, turn_id: str, session_id: str) -> bool:
        from sqlalchemy import select

        row = await session.scalar(select(self._runs.c.id).where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
        ))
        return row is not None

    async def guard(
        self, session: AsyncSession, *, turn_id: str, session_id: str, generation: int,
    ) -> None:
        """Serialize source effects and checkpoint writes with source sealing."""
        columns = self._runs.c
        held = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
            columns.turn_source_generation == generation,
            columns.turn_source_phase == "open",
        ).values(turn_source_generation=columns.turn_source_generation))
        if held.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def latest_complete_checkpoint_key(
        self, session: AsyncSession, *, turn_id: str, session_id: str,
    ) -> str:
        """Read the last occurrence after taking the source write fence.

        Skipping over an interrupted occurrence would resume the wrong model
        frontier. Only a complete, explicitly addressed checkpoint can resume.
        """
        from sqlalchemy import select

        table = self._snapshots
        row = (await session.execute(select(table.c.state, table.c.idempotency_key).where(
            *self._where(table, turn_id=turn_id, session_id=session_id), table.c.pruned.is_(False),
        ).order_by(table.c.id.desc()).limit(1))).one_or_none()
        if row is None or row.state != "complete":
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        try:
            key = decode_native_replay_key(row.idempotency_key)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if key is None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return key

    async def guard_sealed(self, session: AsyncSession, source: SplitStepSourceObservation) -> None:
        """Serialize reconciliation with adoption of this exact sealed generation."""
        columns = self._runs.c
        held = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase == "sealed",
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_generation=columns.turn_source_generation))
        if held.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def seal(
        self, session: AsyncSession, *, turn_id: str, session_id: str,
        generation: int, pin: SplitStepPin,
    ) -> bool:
        """Stop source writes and retain the exact handoff in the same transaction."""
        if pin.waiting is not None:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        return await self._seal(session, turn_id=turn_id, session_id=session_id, generation=generation, pin=pin)

    async def seal_waiting(
        self, session: AsyncSession, *, turn_id: str, session_id: str,
        generation: int, pin: SplitStepPin,
    ) -> bool:
        """Retain only a wait whose exact target frontier can be recovered."""
        from ._session_turn_sql import CanonicalDeferredFrontier, SessionTurnKey

        waiting, checkpoint_id = pin.waiting, pin.checkpoint_id
        if waiting is None or checkpoint_id is None or waiting.original_generation != generation:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        frontier = CanonicalDeferredFrontier.from_split_waiting(
            SessionTurnKey(session_id, turn_id), checkpoint_id=checkpoint_id, waiting=waiting,
        )
        if pin.receipt_id != frontier.split_receipt_id:
            # This source has no owner of arbitrary deferred metadata. Do not
            # commit a wait whose target payload could be lost on process exit.
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        return await self._seal(session, turn_id=turn_id, session_id=session_id, generation=generation, pin=pin)

    async def _seal(
        self, session: AsyncSession, *, turn_id: str, session_id: str,
        generation: int, pin: SplitStepPin,
    ) -> bool:
        from sqlalchemy import select

        payload = pin.to_json()
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
            columns.turn_source_generation == generation,
            columns.turn_source_phase == "open",
        ).values(
            turn_source_phase="sealed" if pin.waiting is None else "waiting",
            turn_source_receipt=pin.receipt_id, turn_source_pin=payload,
        ))
        if changed.rowcount != 1:
            return False
        if pin.checkpoint_id is not None:
            rows = self._snapshots
            query = select(rows.c.id).where(
                *self._where(rows, turn_id=turn_id, session_id=session_id),
                rows.c.idempotency_key == encode_native_replay_key(pin.checkpoint_id),
                rows.c.pruned.is_(False),
            )
            if pin.waiting is not None:
                query = query.where(rows.c.state == "interrupted")
            found = await session.scalar(query)
            if found is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if pin.transcript is not None:
            if pin.transcript.store_key != self.store_key:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            stage_owner(session, self._runs.name, self.store_key, (turn_id,), pin.transcript)
        return True

    async def settle_waiting_cancel(
        self, session: AsyncSession, source: SplitStepSourceObservation, *, generation: int, result: JsonValue,
    ) -> SplitStepSourceObservation | None:
        if (source.phase not in ("waiting", "ready", "resuming", "resume_hold") or source.pin.waiting is None
                or (source.generation != generation and not (
                    source.generation == generation - 1 and source.phase in ("ready", "resume_hold")
                ))):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        columns = self._runs.c
        pin = SplitStepPin(
            source.pin.receipt_id, checkpoint_id=source.pin.checkpoint_id, terminal_state="cancelled", result=result,
        )
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation, columns.turn_source_phase == source.phase,
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_generation=generation, turn_source_phase="sealed", turn_source_pin=pin.to_json()))
        return (SplitStepSourceObservation(source.turn_id, source.session_id, generation, "sealed", pin)
                if changed.rowcount == 1 else None)

    async def settle_sealed_cancel(
        self, session: AsyncSession, source: SplitStepSourceObservation, *, result: JsonValue,
    ) -> SplitStepSourceObservation | None:
        """Settle a nonterminal checkpoint seal after the target records Stop.

        The previous receipt identity and checkpoint remain exact. A terminal
        source (including succeeded) cannot be replaced by this path.
        """
        prior = source.pin
        if (source.phase != "sealed" or prior.waiting is not None
                or prior.terminal_state is not None or prior.checkpoint_id is None):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        pin = SplitStepPin(
            prior.receipt_id, checkpoint_id=prior.checkpoint_id, terminal_state="cancelled", result=result,
        )
        columns = self._runs.c
        # Serialize with competing source adoption and parse exact committed
        # source metadata before modifying it. No JSON equality predicate is
        # needed; the guarded owner row remains locked through publication.
        await self.guard_sealed(session, source)
        committed = await self.observe_pin(
            session, turn_id=source.turn_id, session_id=source.session_id, generation=source.generation,
        )
        if committed != source:
            return committed if committed is not None and committed.pin == pin else None
        await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
        ).values(turn_source_pin=pin.to_json()))
        return SplitStepSourceObservation(source.turn_id, source.session_id, source.generation, "sealed", pin)

    async def mark_waiting_ready(self, session: AsyncSession, source: SplitStepSourceObservation) -> bool:
        """Advance this exact pin inside the transaction settling its existing leaf."""
        if source.phase != "waiting" or source.pin.waiting is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase == "waiting",
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_phase="ready"))
        return changed.rowcount == 1

    async def observe_pin(
        self, session: AsyncSession, *, turn_id: str, session_id: str, generation: int,
    ) -> SplitStepSourceObservation | None:
        """Read back the persisted receipt after source commit or commit ambiguity."""
        from sqlalchemy import select

        columns = self._runs.c
        row = (await session.execute(select(
            columns.turn_source_phase, columns.turn_source_receipt, columns.turn_source_pin,
        ).where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
            columns.turn_source_generation == generation,
            columns.turn_source_phase.in_(("sealed", "published", "discarded", "waiting", "ready", "resuming", "resume_hold")),
        ))).one_or_none()
        if row is None:
            return None
        phase, receipt_id, value = row
        try:
            pin = SplitStepPin.from_json(receipt_id, value)
            if (phase in ("waiting", "ready", "resuming", "resume_hold")) != (pin.waiting is not None):
                raise ValueError("split waiting pin does not match its source phase")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        return SplitStepSourceObservation(turn_id, session_id, generation, phase, pin)

    async def reconcile_discarded(self, session: AsyncSession, source: SplitStepSourceObservation) -> bool:
        """Release an unprojected terminal outcome only after exact target cancellation proof."""
        if source.pin.terminal_state not in ("succeeded", "failed"):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=source.turn_id, session_id=source.session_id),
            columns.turn_source_generation == source.generation,
            columns.turn_source_phase.in_(("sealed", "published")),
            columns.turn_source_receipt == source.pin.receipt_id,
        ).values(turn_source_phase="discarded"))
        if changed.rowcount and source.pin.transcript is not None:
            # Leave the immutable source descriptor/identity for exact receipts.
            # The detached body no longer owns content; ordinary Step retention
            # can still retain any message occurrences referenced elsewhere.
            stage_owner(session, self._runs.name, self.store_key, (source.turn_id,), None)
        return changed.rowcount == 1

    async def reconcile(
        self, session: AsyncSession, *, turn_id: str, session_id: str, generation: int,
        exact_target_receipt: str, target_removed: bool = False,
    ) -> bool:
        """Match a confirmed destination receipt; an elapsed lease proves neither outcome."""
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
            columns.turn_source_generation == generation,
            columns.turn_source_phase == ("published" if target_removed else "sealed"),
            columns.turn_source_receipt == exact_target_receipt,
        ).values(
            **({"turn_source_phase": "retired", "turn_source_pin": None} if target_removed
               else {"turn_source_phase": "published"}),
        ))
        return changed.rowcount == 1
