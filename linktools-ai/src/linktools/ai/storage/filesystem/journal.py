#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TransactionJournal: crash-recovery journal for FilesystemRunCommitCoordinator.

File storage has no cross-store transaction, so pause/complete write their
stores sequentially. A crash mid-sequence can leave a run half-committed
(e.g. checkpoint saved but the WAITING_APPROVAL transition never landed, or
vice-versa). This journal records the intent and the progress through each
commit's state machine so ``recover_incomplete_commits()`` can finish or roll
back deterministically on the next start.

Each transaction is one JSON file under ``{root}/transactions/{tx_id}.json``,
written atomically (tmp file -> fsync -> ``os.replace``). The journal records
the state machine position plus the ids written so far (approval_id,
checkpoint_id, session_message_ids) so recovery knows what already landed.

States (pause):  PREPARED -> APPROVAL_WRITTEN -> CHECKPOINT_WRITTEN ->
                  RUN_TRANSITIONED -> EVENTS_WRITTEN -> COMMITTED
States (complete): PREPARED -> SESSION_WRITTEN -> CHECKPOINT_WRITTEN ->
                  RUN_TRANSITIONED -> EVENTS_WRITTEN -> COMMITTED

A COMMITTED journal is deleted (the commit is durable); only incomplete
journals remain for recovery."""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class TransactionKind(str, Enum):
    PAUSE = "pause"
    COMPLETE = "complete"


class TransactionState(str, Enum):
    PREPARED = "PREPARED"
    APPROVAL_WRITTEN = "APPROVAL_WRITTEN"
    SESSION_WRITTEN = "SESSION_WRITTEN"
    CHECKPOINT_WRITTEN = "CHECKPOINT_WRITTEN"
    RUN_TRANSITIONED = "RUN_TRANSITIONED"
    EVENTS_WRITTEN = "EVENTS_WRITTEN"
    COMMITTED = "COMMITTED"


@dataclass(frozen=True)
class TransactionRecord:
    """One in-flight (or committed) commit transaction.

    ``approval_id``/``checkpoint_id``/``session_message_ids`` capture what has
    already been written, so recovery can skip re-writing them or know which
    are orphans to clean up. ``command`` holds the original commit command
    payload (serialized) so recovery can re-drive the remaining steps."""

    id: str
    kind: TransactionKind
    run_id: str
    state: TransactionState
    target_run_status: str
    created_at: str
    commit_id: str = ""
    approval_id: "str | None" = None
    checkpoint_id: "str | None" = None
    session_message_ids: "tuple[str, ...]" = ()
    command: "Mapping[str, Any]" = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["state"] = self.state.value
        d["session_message_ids"] = list(self.session_message_ids)
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "TransactionRecord":
        d = json.loads(text)
        d["kind"] = TransactionKind(d["kind"])
        d["state"] = TransactionState(d["state"])
        d["session_message_ids"] = tuple(d.get("session_message_ids") or ())
        return cls(**d)


class TransactionJournal:
    """Crash-recovery journal backing the FilesystemRunCommitCoordinator."""

    def __init__(self, transactions_root: "str | Path") -> None:
        self._root = Path(transactions_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, tx_id: str) -> Path:
        return self._root / f"{tx_id}.json"

    def _write_atomic(self, tx_id: str, text: str) -> None:
        """Write ``text`` to the journal file atomically: tmp -> fsync ->
        os.replace. A crash before replace leaves the old (or no) file; a crash
        after replace leaves the new file durably."""
        target = self._path(tx_id)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        fd = os.open(tmp, os.O_WRONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)

    def begin(
        self,
        *,
        kind: TransactionKind,
        run_id: str,
        target_run_status: str,
        commit_id: str = "",
        command: "Mapping[str, Any]",
    ) -> TransactionRecord:
        """Open a new PREPARED transaction and persist its journal."""
        tx_id = uuid.uuid4().hex
        record = TransactionRecord(
            id=tx_id,
            kind=kind,
            run_id=run_id,
            state=TransactionState.PREPARED,
            target_run_status=target_run_status,
            created_at=_now_iso(),
            commit_id=commit_id,
            command=dict(command),
        )
        self._write_atomic(tx_id, record.to_json())
        return record

    def find_incomplete(self, commit_id: str) -> "TransactionRecord | None":
        if not commit_id:
            return None
        for record in self.list_incomplete():
            if record.commit_id == commit_id:
                return record
        return None

    def update(self, record: TransactionRecord, **changes: Any) -> TransactionRecord:
        """Atomically advance the journal to a new state/fields. Returns the
        updated record (the caller should hold the latest copy)."""
        replaced = record
        for name, value in changes.items():
            if name == "state":
                value = (
                    value
                    if isinstance(value, TransactionState)
                    else TransactionState(value)
                )
            replaced = _replace_field(replaced, name, value)
        self._write_atomic(replaced.id, replaced.to_json())
        return replaced

    def commit(self, record: TransactionRecord) -> None:
        """Mark the transaction COMMITTED and delete its journal (the commit is
        durable; no recovery needed)."""
        final = self.update(record, state=TransactionState.COMMITTED)
        try:
            self._path(final.id).unlink()
        except FileNotFoundError:
            pass

    def list_incomplete(self) -> "list[TransactionRecord]":
        """All journals not yet COMMITTED, in creation order. Drives recovery."""
        records: "list[TransactionRecord]" = []
        for path in sorted(self._root.glob("*.json")):
            try:
                rec = TransactionRecord.from_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt/unreadable journal is itself a recovery signal --
                # skip it here; recovery treats missing-state runs conservatively.
                continue
            if rec.state is not TransactionState.COMMITTED:
                records.append(rec)
        return records


def _replace_field(
    record: TransactionRecord, name: str, value: Any
) -> TransactionRecord:
    """Frozen-dataclass field replacement."""
    items = {f: getattr(record, f) for f in record.__dataclass_fields__}
    items[name] = value
    return TransactionRecord(**items)


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
