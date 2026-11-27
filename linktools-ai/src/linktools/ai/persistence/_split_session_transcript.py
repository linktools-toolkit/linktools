#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read canonical transcript references pinned in the configured Step source."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..steps import native_store_locator
from ._message_root import NativeMessageRoot
from .state import SqlStateRowReader

if TYPE_CHECKING:
    from .state import StateStore


@dataclass(frozen=True, slots=True)
class SplitTranscriptReference:
    """Only construct from an authorized, succeeded Conversation turn."""

    turn_id: str
    session_id: str
    generation: int
    receipt_id: str
    root: NativeMessageRoot
    terminal_text_bytes: int


class SplitSessionTranscriptReader:
    """Bound to a single configured source, never routed from a wire group ID."""

    def __init__(
        self, archive_store: StateStore, *, group_id: str, store_key: bytes, capture: Callable[..., Awaitable],
    ) -> None:
        self._store = archive_store
        self._group = archive_store.storage_group
        self._runs = self._group.metadata.tables["ai_step_runs"]
        self._group_id, self._store_key, self._capture = group_id, store_key, capture

    async def read(
        self, references: Sequence[SplitTranscriptReference],
    ) -> tuple[tuple[bytes, ...], ...]:
        from sqlalchemy import select, tuple_

        if not references:
            return ()
        expected = {}
        for ref in references:
            if ref.root.group_id != self._group_id or ref.root.store_key != self._store_key:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            key = ref.turn_id, ref.session_id, ref.generation, ref.receipt_id
            value = ref.root, ref.terminal_text_bytes
            if expected.setdefault(key, value) != value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def capture(reader):
            if not isinstance(reader, SqlStateRowReader):
                raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
            cols = self._runs.c
            identity = cols.run_id, cols.conversation_id, cols.turn_source_generation, cols.turn_source_receipt
            keys = tuple(expected)
            for offset in range(0, len(keys), 128):
                batch = keys[offset:offset + 128]
                statement = select(*identity, cols.turn_source_pin).where(
                    cols.store_key == self._store_key,
                    cols.store_locator == native_store_locator(self._store_key),
                    cols.turn_source_phase.in_(("sealed", "published")),
                    tuple_(*identity).in_(batch),
                )
                found = await reader.read_sql_rows(self._group.context, statement)
                observed = set()
                for row in found:
                    receipt = tuple(row[column.key] for column in identity)
                    pin = row["turn_source_pin"]
                    try:
                        if (not isinstance(pin, Mapping) or pin.get("terminal_state") != "succeeded"
                                or type(pin.get("terminal_text_bytes")) is not int
                                or (NativeMessageRoot.from_json(pin["transcript"]), pin["terminal_text_bytes"])
                                != expected[receipt]):
                            raise ValueError("source pin differs from the committed transcript")
                    except (KeyError, TypeError, ValueError, AIError) as error:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    observed.add(receipt)
                if observed != set(batch):
                    # Conversation read and source read are distinct SQL
                    # snapshots. Deletion racing the former may retire a pin
                    # before this snapshot; never substitute another root.
                    raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            return await self._capture(tuple(ref.root for ref in references), transaction=reader)

        load = await self._store.read(capture)
        return await load()
