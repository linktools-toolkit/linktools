#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline reclamation of native content after its snapshot roots retire."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import AIError, ErrorCode
from ._archive import ImmutableSequenceNode

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


def _reachable_content(
    snapshots: Sequence[Mapping[str, object]],
    rows: Mapping[str, Mapping[str, object]],
) -> tuple[set[str], set[str]]:
    reachable: set[str] = set()
    messages: set[str] = set()
    checked: dict[str, ImmutableSequenceNode] = {}

    def node(identity: str, count: int) -> ImmutableSequenceNode:
        value = checked.get(identity)
        if value is None:
            row = rows.get(identity)
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = ImmutableSequenceNode(
                row["message_sha256"], row["tree_count"],
                row["left_sha256"], row["right_sha256"], row["next_sha256"],
            )
            if value.identity != identity:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            checked[identity] = value
        if value.tree_count != count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    for snapshot in snapshots:
        root = snapshot["sequence_root"]
        if snapshot["pruned"]:
            if root is not None or snapshot["messages"] != "{}":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            continue
        try:
            descriptor = json.loads(snapshot["messages"])
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if not isinstance(descriptor, dict) or set(descriptor) != {"stored_messages", "checkpoint"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        count = descriptor["stored_messages"]
        if type(count) is not int or count < 0 or (root is None) != (count == 0):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        first = root
        for weight in ImmutableSequenceNode.forest_weights(count):
            head = node(first, weight)
            pending = [(first, weight)]
            while pending:
                identity, size = pending.pop()
                value = node(identity, size)
                if identity in reachable:
                    continue
                reachable.add(identity)
                messages.add(value.content_id)
                if size > 1:
                    child_size = (size - 1) // 2
                    pending.extend(((value.left, child_size), (value.right, child_size)))
            # A subtree's historical next pointer is not part of this tree's
            # visible sequence. Only forest heads continue the captured root.
            first = head.next_forest
        if first is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return reachable, messages


async def purge_step_content(
    context: "SqlStorageContext", metadata: "MetaData", store_digest: str,
) -> StepContentCleanupResult:
    """Reclaim one offline namespace in one rollback-safe owner transaction.

    Three scoped metadata reads precede all deletion. Bodies are not downloaded
    or decoded. This is a complete offline scan, not bounded online maintenance.
    """
    from sqlalchemy import delete, select

    snapshots = metadata.tables["ai_step_snapshots"]
    sequences = metadata.tables["ai_step_sequences"]
    messages = metadata.tables["ai_step_messages"]

    async def mutate(session: "AsyncSession") -> StepContentCleanupResult:
        roots = (await session.execute(select(
            snapshots.c.sequence_root, snapshots.c.messages, snapshots.c.pruned,
        ).where(snapshots.c.store_digest == store_digest))).mappings().all()
        rows = (await session.execute(select(
            sequences.c.sha256, sequences.c.message_sha256, sequences.c.tree_count,
            sequences.c.left_sha256, sequences.c.right_sha256, sequences.c.next_sha256,
        ).where(sequences.c.store_digest == store_digest))).mappings().all()
        by_identity = {row["sha256"]: row for row in rows}
        retained_nodes, retained_messages = _reachable_content(roots, by_identity)
        body_ids = set((await session.execute(select(messages.c.sha256).where(
            messages.c.store_digest == store_digest,
        ))).scalars().all())
        if not retained_messages.issubset(body_ids):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        unused_nodes = sorted(set(by_identity).difference(retained_nodes))
        unused_messages = sorted(body_ids.difference(retained_messages))
        for table, identities in ((sequences, unused_nodes), (messages, unused_messages)):
            for offset in range(0, len(identities), 128):
                batch = identities[offset:offset + 128]
                removed = await session.execute(delete(table).where(
                    table.c.store_digest == store_digest, table.c.sha256.in_(batch),
                ))
                if removed.rowcount != len(batch):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
        return StepContentCleanupResult(
            len(unused_nodes), len(unused_messages), len(retained_nodes), len(retained_messages),
        )

    return await context.run_mutation(mutate, domain="steps.content_cleanup")


__all__ = ["StepContentCleanupResult"]
