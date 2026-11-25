#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subordinate SQL fences for a canonical turn's separately stored Step run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import JsonValue, validate_resource_id
from ..errors import AIError, ErrorCode
from ..steps import decode_native_replay_key, encode_native_replay_key, native_store_locator
from ._message_root import NativeMessageRoot
from ._native_content_index import stage_owner

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import Table


@dataclass(frozen=True, slots=True)
class SplitStepPin:
    receipt_id: str
    checkpoint_id: str | None = None
    terminal_state: str | None = None
    transcript: NativeMessageRoot | None = None
    result: JsonValue = None
    binding_refs: tuple[tuple[int, int | None], ...] = ()
    output_ref: tuple[int, int] | None = None

    def to_json(self) -> dict:
        validate_resource_id(self.receipt_id)
        if len(self.receipt_id.encode("utf-8")) > 64:
            raise ValueError("split source receipt is too wide")
        if self.checkpoint_id is not None:
            encode_native_replay_key(self.checkpoint_id)
        if self.terminal_state not in (None, "succeeded", "failed", "cancelled"):
            raise ValueError("invalid split terminal intent")
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
            "transcript": None if self.transcript is None else self.transcript.to_json(),
        }


    @classmethod
    def from_json(cls, receipt_id: str, value: dict) -> SplitStepPin:
        if value.keys() != {"checkpoint_id", "terminal_state", "transcript", "result", "binding_refs", "output_ref"}:
            raise ValueError("invalid split source pin")
        transcript = value["transcript"]
        output_ref = value["output_ref"]
        pin = cls(
            receipt_id, checkpoint_id=value["checkpoint_id"], terminal_state=value["terminal_state"],
            transcript=None if transcript is None else NativeMessageRoot.from_json(transcript),
            result=value["result"], binding_refs=tuple(tuple(ref) for ref in value["binding_refs"]),
            output_ref=None if output_ref is None else tuple(output_ref),
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
        columns = self._runs.c
        previous = ((columns.turn_source_generation.is_(None),) if predecessor is None else (
            columns.turn_source_generation == generation - 1,
            columns.turn_source_phase == "sealed",
            columns.turn_source_receipt == predecessor.receipt_id,
        ))
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
        from sqlalchemy import select

        payload = pin.to_json()
        columns = self._runs.c
        changed = await session.execute(self._runs.update().where(
            *self._where(self._runs, turn_id=turn_id, session_id=session_id),
            columns.turn_source_generation == generation,
            columns.turn_source_phase == "open",
        ).values(
            turn_source_phase="sealed", turn_source_receipt=pin.receipt_id, turn_source_pin=payload,
        ))
        if changed.rowcount != 1:
            return False
        if pin.checkpoint_id is not None:
            rows = self._snapshots
            found = await session.scalar(select(rows.c.id).where(
                *self._where(rows, turn_id=turn_id, session_id=session_id),
                rows.c.idempotency_key == encode_native_replay_key(pin.checkpoint_id),
                rows.c.pruned.is_(False),
            ))
            if found is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if pin.transcript is not None:
            if pin.transcript.store_key != self.store_key:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            stage_owner(session, self._runs.name, self.store_key, (turn_id,), pin.transcript)
        return True

    async def settle_sealed_cancel(
        self, session: AsyncSession, source: SplitStepSourceObservation, *, result: JsonValue,
    ) -> SplitStepSourceObservation | None:
        """Settle a nonterminal checkpoint seal after the target records Stop.

        The previous receipt identity and checkpoint remain exact. A terminal
        source (including succeeded) cannot be replaced by this path.
        """
        prior = source.pin
        if source.phase != "sealed" or prior.terminal_state is not None or prior.checkpoint_id is None:
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
            columns.turn_source_phase.in_(("sealed", "published", "discarded")),
        ))).one_or_none()
        if row is None:
            return None
        phase, receipt_id, value = row
        try:
            pin = SplitStepPin.from_json(receipt_id, value)
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
