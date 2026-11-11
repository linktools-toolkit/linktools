#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline reclamation of native content after its snapshot roots retire."""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ..model.attachments import read_attachment_payload
from ..storage import ObjectRef
from ._archive_closure import (
    NativeMessageClosure, iter_native_message_roots, require_known_retention,
    validate_native_message_content,
)
from ._capability_marker import has_capability_partition_marker
from ._message_root import NativeMessageRoot
from ._payload import decode_payload_bytes
from .state import FactScanCursor, OperationScanCursor, RecordScanCursor, StateStorageGroup

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncSession

    from ..storage import SqlStorageContext


@dataclass(frozen=True, slots=True)
class StepContentCleanupResult:
    """Confirmed native content counts; snapshot and replay identities remain."""

    deleted_sequences: int
    deleted_messages: int
    retained_sequences: int
    retained_messages: int


async def inspect_native_message_tables(
    context: "SqlStorageContext", expected: "MetaData", *, require_content: bool = False,
) -> tuple[str, ...]:
    """Resolve optional snapshot ownership without accepting broken native pairs.

    Read the physical schema, not just active routes: a checkpoint owner outside
    this adapter's selected domains must still contribute every retained root.
    Conversely, metadata-required tables cannot disappear silently.
    """
    from sqlalchemy import inspect

    names = ("ai_step_snapshots", "ai_step_sequences", "ai_step_messages")
    async with context.engine.connect() as connection:
        present = await connection.run_sync(
            lambda connection: tuple(inspect(connection).has_table(name) for name in names),
        )
    actual = {name for name, exists in zip(names, present) if exists}
    required = set(names).intersection(expected.tables)
    content = {"ai_step_sequences", "ai_step_messages"}
    if require_content:
        required.update(content)
    if (not required.issubset(actual)
            or bool(actual.intersection(content)) != content.issubset(actual)
            or "ai_step_snapshots" in actual and not content.issubset(actual)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return tuple(name for name in names if name in actual)


async def scan_native_message_roots(group: "StateStorageGroup") -> tuple[NativeMessageRoot, ...]:
    """Scan all durable partitions under the caller's offline exclusion."""
    from .captures import CAPTURE_RECORD_KINDS, capture_retention_value

    roots: list[NativeMessageRoot] = []
    for store in await group.scan_stores():
        if await store.read(has_capability_partition_marker):
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
        cursor = None
        while True:
            page = await store.read(lambda tx: tx.scan_records_page(after=cursor, limit=1000))
            for row in page:
                projected = (None if row.kind == "archive_content" else
                             capture_retention_value(row) if row.kind in CAPTURE_RECORD_KINDS else row.data)
                require_known_retention(projected, state=row.state)
                roots.extend(iter_native_message_roots(projected))
            if len(page) < 1000:
                break
            cursor = RecordScanCursor(page[-1].kind, page[-1].key)
        cursor = None
        while True:
            page = await store.read(lambda tx: tx.scan_facts_page(after=cursor, limit=1000))
            for row in page:
                require_known_retention(row.data, state=row.state)
                roots.extend(iter_native_message_roots(row.data))
            if len(page) < 1000:
                break
            cursor = FactScanCursor(page[-1].stream_digest, page[-1].sequence)
        cursor = None
        while True:
            page = await store.read(lambda tx: tx.scan_operations_page(after=cursor, limit=1000))
            for row in page:
                require_known_retention(row.data, state=row.state)
                roots.extend(iter_native_message_roots(row.data))
            if len(page) < 1000:
                break
            cursor = OperationScanCursor(page[-1].key)
    return tuple(roots)


async def purge_step_content(
    context: "SqlStorageContext", metadata: "MetaData", store_digest: str, *,
    group: StateStorageGroup | None = None,
    read_object: Callable[[ObjectRef], Awaitable[bytes]] | None = None,
) -> StepContentCleanupResult:
    """Reclaim one offline namespace in one rollback-safe owner transaction.

    All retained owner references and canonical content are verified before
    deletion. Body reads are limited to batches of retained messages; identity
    metadata and owner roots still require a complete offline scan. This scan
    never runs in normal reads or writes.
    """
    from sqlalchemy import delete, inspect, select

    from ._sql import SqlStateStorageGroup

    if group is not None and (not isinstance(group, SqlStateStorageGroup)
                              or group.context.engine is not context.engine):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if group is None:
        # An unbound adapter must not mistake a shared Runtime database for a
        # standalone StepStore and omit its durable cross-owner references.
        async with context.engine.connect() as connection:
            has_state = await connection.run_sync(lambda connection: any(
                inspect(connection).has_table(name)
                for name in ("ai_state_records", "ai_state_facts", "ai_state_operations")
            ))
        if has_state:
            raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING,
                          "Shared native content cleanup requires its bound state group")
    group_id = None
    owner_roots = () if group is None else await scan_native_message_roots(group)
    if group is not None:
        opened = await group.open_archive()
        if opened is None or any(root.group_id != opened[0] for root in owner_roots):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        group_id = opened[0]

    tables = await inspect_native_message_tables(
        context, metadata if group is None else group.metadata, require_content=True,
    )
    snapshots = metadata.tables["ai_step_snapshots"] if "ai_step_snapshots" in tables else None
    sequences = metadata.tables["ai_step_sequences"]
    messages = metadata.tables["ai_step_messages"]

    async def mutate(session: "AsyncSession") -> StepContentCleanupResult:
        roots = [] if snapshots is None else (await session.execute(select(
            snapshots.c.store_digest, snapshots.c.sequence_root, snapshots.c.messages, snapshots.c.pruned,
        ).where(snapshots.c.store_digest == store_digest))).mappings().all()
        rows = (await session.execute(select(
            sequences.c.store_digest, sequences.c.sha256, sequences.c.message_sha256, sequences.c.tree_count,
            sequences.c.left_sha256, sequences.c.right_sha256, sequences.c.next_sha256,
        ).where(sequences.c.store_digest == store_digest))).mappings().all()
        closure = NativeMessageClosure(rows)
        for snapshot in roots:
            closure.mark_snapshot(snapshot)
        for root in owner_roots:
            if root.store_digest == store_digest:
                closure.mark(store_digest, root.slices)
        # Candidate discovery reads identities only: orphan bodies may dominate
        # the namespace's bytes and are not needed to prove retained integrity.
        body_ids = set((await session.execute(select(messages.c.sha256).where(
            messages.c.store_digest == store_digest,
        ))).scalars())
        retained_nodes = {identity for _, identity in closure.retained_sequences}
        retained_messages = {identity for _, identity in closure.retained_messages}
        if not retained_messages.issubset(body_ids):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        retained_ids = sorted(retained_messages)
        for offset in range(0, len(retained_ids), 128):
            batch = retained_ids[offset:offset + 128]
            body_rows = (await session.execute(select(
                messages.c.store_digest, messages.c.sha256, messages.c.message,
                messages.c.compression, messages.c.raw_size, messages.c.payload_json,
            ).where(
                messages.c.store_digest == store_digest, messages.c.sha256.in_(batch),
            ))).mappings().all()
            if {row["sha256"] for row in body_rows} != set(batch):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for row in body_rows:
                content = validate_native_message_content(row, group_id=group_id)
                if content.payload is not None:
                    if read_object is None:
                        raise AIError(ErrorCode.STORAGE_CAPABILITY_MISSING)
                    encoded = await read_attachment_payload(content.payload, read_object=read_object)
                    if len(encoded) != content.payload.size or hashlib.sha256(encoded).hexdigest() != content.payload.digest:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    decode_payload_bytes(encoded, codec=content.compression,
                                         raw_digest=content.raw_digest, raw_size=content.raw_size)
        unused_nodes = sorted({row["sha256"] for row in rows}.difference(retained_nodes))
        unused_messages = sorted(body_ids.difference(retained_messages))
        for table, identities in ((sequences, unused_nodes), (messages, unused_messages)):
            for offset in range(0, len(identities), 128):
                batch = identities[offset:offset + 128]
                from ._native_content_index import replace_owner_edges
                if table is sequences:
                    sources = select(table.c.id).where(table.c.store_digest == store_digest, table.c.sha256.in_(batch))
                    await replace_owner_edges(context, session, "sequence", sources, ())
                    await replace_owner_edges(context, session, "message", sources, ())
                edge_table = metadata.tables["ai_native_content_edges"]
                target_ids = select(table.c.id).where(table.c.store_digest == store_digest, table.c.sha256.in_(batch))
                edge_kinds = (("record", "fact", "operation", "snapshot", "sequence", "candidate_sequence")
                              if table is sequences else ("message", "candidate_message"))
                await session.execute(delete(edge_table).where(
                    edge_table.c.edge_kind.in_(edge_kinds), edge_table.c.target_id.in_(target_ids),
                ))
                removed = await session.execute(delete(table).where(
                    table.c.store_digest == store_digest, table.c.sha256.in_(batch),
                ))
                if removed.rowcount != len(batch):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        return StepContentCleanupResult(
            len(unused_nodes), len(unused_messages), len(retained_nodes), len(retained_messages),
        )

    return await context.run_mutation(mutate, domain="steps.content_cleanup")


__all__ = ["StepContentCleanupResult", "scan_native_message_roots", "inspect_native_message_tables"]
