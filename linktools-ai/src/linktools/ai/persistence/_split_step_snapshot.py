#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact terminal pairing for source-first restore of canonical split Step."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..core import validate_resource_id
from ..errors import AIError, ErrorCode
from ..steps import encode_native_replay_key, native_store_locator
from ._message_root import NativeMessageRoot
from ._native_content_index import stage_owner
from ._plan import PersistenceDomain
from ._session_transcript import decode_session_transcript_root

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import Table
    from ._step_backup import NativeStepBackup
    from .state import SqlStateRowReader
    from ..storage import SqlStorageContext


from ._split_step_source import SplitStepPin

SPLIT_SOURCE_COLUMNS = ("turn_source_generation", "turn_source_phase", "turn_source_receipt", "turn_source_pin")


async def require_no_unpaired_split_pins(
    reader: SqlStateRowReader, context: SqlStorageContext, table: Table, store_key: bytes,
) -> None:
    """A portable owner read may not silently drop a live or retained pin."""
    from sqlalchemy import or_, select

    columns = table.c
    if not all(name in columns for name in SPLIT_SOURCE_COLUMNS):
        return
    rows = await reader.read_sql_rows(context, select(columns.id).where(
        columns.store_locator == native_store_locator(store_key), columns.store_key == store_key,
        or_(*(columns[name].is_not(None) for name in SPLIT_SOURCE_COLUMNS)),
    ).limit(1))
    if rows:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Split Step source requires a paired snapshot")


async def capture_split_source_pins(
    reader: SqlStateRowReader, context: SqlStorageContext, table: Table, store_key: bytes,
    *, accept: Callable[[object], object],
) -> tuple[dict[str, object], ...]:
    """Capture the entire scoped pin inventory in the native group's read view."""
    from sqlalchemy import or_, select

    columns = table.c
    if not all(name in columns for name in SPLIT_SOURCE_COLUMNS):
        return ()
    output = []
    after = 0
    while True:
        rows = await reader.read_sql_rows(context, select(
            columns.id, columns.run_id, columns.conversation_id, *(columns[name] for name in SPLIT_SOURCE_COLUMNS),
        ).where(
            columns.store_locator == native_store_locator(store_key), columns.store_key == store_key,
            columns.id > after, or_(*(columns[name].is_not(None) for name in SPLIT_SOURCE_COLUMNS)),
        ).order_by(columns.id).limit(128))
        for row in rows:
            if row["turn_source_phase"] in ("waiting", "ready", "resuming", "resume_hold"):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Deferred split Step source requires a nonterminal snapshot")
            if row["turn_source_phase"] not in ("sealed", "published", "discarded"):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED, "Live or unresolved split Step source cannot be exported")
            try:
                pin = SplitStepPin.from_json(row["turn_source_receipt"], row["turn_source_pin"])
                if pin.terminal_state is None:
                    raise ValueError("non-terminal source pin")
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            captured = {
                "turn_id": row["run_id"], "session_id": row["conversation_id"],
                "generation": row["turn_source_generation"], "phase": row["turn_source_phase"],
                "receipt_id": pin.receipt_id, "pin": pin.to_json(),
            }
            accept(captured)
            output.append(captured)
        if len(rows) < 128:
            return tuple(output)
        after = rows[-1]["id"]


@dataclass(frozen=True, slots=True)
class _SplitTerminalPair:
    turn_id: str
    session_id: str
    generation: int
    receipt_id: str
    pin: SplitStepPin
    target_succeeded: bool
    detached: bool

    @property
    def root(self) -> NativeMessageRoot | None:
        return self.pin.transcript


@dataclass(frozen=True, slots=True)
class PairedSplitTerminalRestore:
    source_group: str
    store_key: bytes
    pairs: tuple[_SplitTerminalPair, ...]

    @classmethod
    def decode(
        cls, wire: object, ordinary: Mapping[str, Sequence[Mapping[str, object]]] | None,
        backups: Mapping[str, NativeStepBackup], entries: Mapping[str, Mapping[str, object]],
        *, conversation_group: str | None,
    ) -> PairedSplitTerminalRestore | None:
        """Require a bijection to exact terminal Conversation receipts before writes."""
        if wire is None:
            return None
        try:
            if not isinstance(wire, Mapping) or wire.keys() != {"source_group", "pins"} or ordinary is None:
                raise ValueError("invalid split backup section")
            source = wire["source_group"]
            pins = wire["pins"]
            if (not isinstance(source, str) or not source or source == conversation_group
                    or not isinstance(pins, list) or not pins):
                raise ValueError("invalid split backup group")
            backup = backups[source]
            if entries[source]["domain"] != PersistenceDomain.EXECUTION.value:
                raise ValueError("split native owner must retain its Execution route")

            targets = {row["turn_id"].decode("utf-8"): row for row in ordinary["turns"]
                       if row.get("split_source_receipt") is not None}
            registrations = {row["run_id"]: row for row in backup.runs}
            checkpoints = {(row["run_id"], row["conversation_id"], row["idempotency_key"])
                           for row in backup.snapshots if not row["pruned"]}
            matched = set()
            pairs = []
            for item in pins:
                if not isinstance(item, Mapping) or item.keys() != {
                    "turn_id", "session_id", "generation", "phase", "receipt_id", "pin",
                }:
                    raise ValueError("invalid split source pin")
                turn_id = validate_resource_id(item["turn_id"])
                session_id = validate_resource_id(item["session_id"])
                receipt_id = validate_resource_id(item["receipt_id"])
                generation = item["generation"]
                pin = SplitStepPin.from_json(receipt_id, item["pin"])
                state = pin.terminal_state
                if pin.waiting is not None:
                    raise ValueError("waiting split source cannot be restored by a terminal snapshot")
                if (turn_id in matched or len(receipt_id.encode("utf-8")) > 64
                        or type(generation) is not int or generation < 1
                        or state not in ("succeeded", "failed", "cancelled")
                        or item["phase"] not in ("sealed", "published", "discarded")):
                    raise ValueError("ambiguous or live split source")
                target = targets[turn_id]
                discard = target["state"] == "cancelled" and state in ("succeeded", "failed")
                detached = item["phase"] == "discarded"
                if detached and not discard:
                    raise ValueError("detached source body must be absent from target publication")
                if (target["session_key"] != session_id.encode("utf-8")
                        or target["generation"] != generation or (target["state"] != state and not discard)
                        or target["split_source_receipt"] != receipt_id.encode("utf-8")
                        or registrations[turn_id]["conversation_id"] != session_id):
                    raise ValueError("split source does not match exact target")
                checkpoint = pin.checkpoint_id
                if checkpoint is not None and (
                    turn_id, session_id, encode_native_replay_key(checkpoint)
                ) not in checkpoints:
                    raise ValueError("split checkpoint is not retained in source backup")
                root = pin.transcript
                if state == "succeeded":
                    if root is None or root.group_id != source or root.store_key != backup.store_key:
                        raise ValueError("split transcript differs from the source group")
                    if not discard and (root != decode_session_transcript_root(target["result_json"])
                            or target["result_json"]["binding_refs"] != [list(ref) for ref in pin.binding_refs]
                            or target["result_json"]["terminal_text_bytes"] != pin.terminal_text_bytes
                            or target["result_json"].get("output_ref") != (
                                None if pin.output_ref is None else list(pin.output_ref)
                            )):
                        raise ValueError("split transcript differs from source pin")
                elif root is not None or (not discard and target["result_json"] != pin.result):
                    raise ValueError("non-success split receipt differs from committed terminal result")
                matched.add(turn_id)
                pairs.append(_SplitTerminalPair(
                    turn_id, session_id, generation, receipt_id, pin, target["state"] == "succeeded", detached,
                ))
            if matched != set(targets):
                raise ValueError("split source and terminal target inventory differ")
            for target in ordinary["turns"]:
                if target["state"] == "succeeded":
                    root = decode_session_transcript_root(target["result_json"])
                    if root.group_id == source and target["turn_id"].decode("utf-8") not in matched:
                        raise ValueError("remote transcript is missing its exact target receipt")
        except (ValueError, KeyError, TypeError, AttributeError, AIError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Invalid paired split Step snapshot") from error
        return cls(source, backup.store_key, tuple(pairs))

    @property
    def roots(self) -> tuple[NativeMessageRoot, ...]:
        return tuple(pair.root for pair in self.pairs if pair.root is not None and not pair.detached)

    @property
    def projected_roots(self) -> tuple[NativeMessageRoot, ...]:
        return tuple(pair.root for pair in self.pairs if pair.root is not None and pair.target_succeeded)

    async def install_source(
        self, session: AsyncSession, table: Table, *, target_group: str,
    ) -> None:
        """Install pins with native data, before any target turn references them.

        Restored evidence is sealed until the new destination is published.
        A failed target commit leaves an independently retained source, never
        an exposed turn without its exact immutable content.
        """
        columns = table.c
        for row in self.pairs:
            root = None if row.root is None else replace(row.root, group_id=target_group)
            changed = await session.execute(table.update().where(
                columns.store_locator == native_store_locator(self.store_key), columns.store_key == self.store_key,
                columns.run_id == row.turn_id, columns.conversation_id == row.session_id,
                columns.turn_source_generation.is_(None), columns.turn_source_phase.is_(None),
                columns.turn_source_receipt.is_(None), columns.turn_source_pin.is_(None),
            ).values(
                turn_source_generation=row.generation, turn_source_phase="discarded" if row.detached else "sealed",
                turn_source_receipt=row.receipt_id,
                turn_source_pin=replace(row.pin, transcript=root).to_json(),
            ))
            if changed.rowcount != 1:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if root is not None and not row.detached:
                stage_owner(session, table.name, self.store_key, (row.turn_id,), root)
