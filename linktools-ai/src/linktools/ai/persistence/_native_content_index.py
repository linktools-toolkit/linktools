#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transactional, rebuildable indexes of native content's actual owners."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, TypeVar

from ..errors import AIError, ErrorCode
from ._step_schema import build_step_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession
    from ..storage import SqlStorageContext


ValueT = TypeVar("ValueT")
_EMPTY = 0
_STATE_KINDS = ("record", "fact", "operation", "snapshot")
_SEQUENCE_EDGES = (*_STATE_KINDS, "sequence")
_MESSAGE_EDGES = ("message",)
_PENDING = "linktools.native_owner_changes"
_RECOVER = "linktools.native_recovery"
_PUBLISHED = "linktools.native_published"
_CLAIMED = "linktools.native_claimed"
_WORK = "linktools.native_gc_work"
_PUBLICATION_HANDOFF = "linktools.native_publication_handoff"


class PublicationTargetsUnavailable(ValueError):
    """A receipt cannot supply IDs in the requested publication scope."""


class _PublicationTargets(Protocol):
    def __call__(
        self, session: "AsyncSession", *, store_digest: str, roots: Sequence[str],
    ) -> tuple[int, ...]: ...


def register_publication_targets(
    session: "AsyncSession", store: str, roots: Sequence[str], resolve: _PublicationTargets,
) -> None:
    """Hand one receipt's live evidence to its pending durable owner."""
    resolve(session, store_digest=store, roots=roots)
    transaction = session.get_nested_transaction() or session.get_transaction()
    current = session.info.get(_PUBLICATION_HANDOFF)
    if current is None or current[0] is not transaction:
        current = (transaction, {})
        session.info[_PUBLICATION_HANDOFF] = current
    current[1][store, tuple(sorted(set(roots)))] = resolve


async def _owner_targets(session: "AsyncSession", store: str, roots: Sequence[str]) -> tuple[int, ...]:
    current = session.info.get(_PUBLICATION_HANDOFF)
    if current is not None:
        resolve = current[1].pop((store, tuple(sorted(set(roots)))), None)
        if resolve is not None:
            try:
                return resolve(session, store_digest=store, roots=roots)
            except PublicationTargetsUnavailable:
                # A completed savepoint or prior transaction is not evidence
                # for the owner's current write. Resolve its actual rows again.
                pass
    return tuple((await resolve_targets(session, "sequence", store, roots)).values())


class NativeContentUnavailable(Exception):
    """A dependency disappeared before its new owner could commit."""

    def __init__(self, recoveries: Sequence[Callable[[], Awaitable[None]]] = ()) -> None:
        super().__init__("Native content publication requires fresh materialization")
        self.recoveries = tuple(recoveries)


def stage_owner(
    session: "AsyncSession", kind: str, store: str, identity: Sequence[str | int | bytes],
    data: object | None, *, state: object = None, inserting: bool = False,
) -> None:
    from ._archive_closure import iter_native_message_roots

    roots = () if data is None else tuple(iter_native_message_roots(data))
    if inserting and not roots:
        return
    session.info.setdefault(_PENDING, {})[(kind, store, tuple(identity))] = roots


def register_recovery(session: "AsyncSession", recovery: Callable[[], Awaitable[None]]) -> None:
    session.info.setdefault(_RECOVER, set()).add(recovery)


async def run_native_mutation(
    context: "SqlStorageContext", callback: Callable[["AsyncSession"], Awaitable[ValueT]], *, domain: str,
) -> tuple[ValueT, bool]:
    maintenance = False

    async def execute(session: "AsyncSession") -> ValueT:
        nonlocal maintenance
        try:
            result = await callback(session)
            await finish_publications(context, session)
            maintenance = bool(session.info.get(_WORK) or session.info.get(_PUBLISHED))
            return result
        except NativeContentUnavailable as error:
            raise NativeContentUnavailable(tuple(session.info.get(_RECOVER, ()))) from error
        finally:
            session.info.pop(_PUBLICATION_HANDOFF, None)

    recovered: set[Callable[[], Awaitable[None]]] = set()
    while True:
        try:
            return await context.run_mutation(execute, domain=domain), maintenance
        except NativeContentUnavailable as error:
            # Only a confirmed rollback reaches here. Each candidate may be
            # rematerialized once, even if later publishers reveal another stale
            # root on the next attempt. Unknown commits never enter this branch.
            fresh = tuple(recover for recover in error.recoveries if recover not in recovered)
            if not fresh:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            for recover in fresh:
                recovered.add(recover)
                await recover()


async def maintain_native_content(context: "SqlStorageContext", *, store_digests: Sequence[str]) -> None:
    from linktools.core import environ
    try:
        await context.run_mutation(lambda session: collect_native_content(context, session, store_digests=store_digests),
                                   domain="native.content_cleanup")
    except Exception:
        # The owner has already committed. Durable candidates survive both a
        # failed collector and an unknown collector commit for the next boundary.
        environ.get_logger("ai.persistence.native").warning("Native content maintenance deferred", exc_info=True)


@lru_cache(maxsize=1)
def _tables():
    metadata = build_step_sql_metadata()
    return metadata.tables["ai_native_content_edges"], metadata.tables["ai_step_sequences"], metadata.tables["ai_step_messages"]


async def _enqueue_select(context: "SqlStorageContext", session: "AsyncSession", selection: "Select") -> None:
    edges, _, _ = _tables()
    # INSERT SELECT is a current DML read on MySQL, unlike a plain SELECT in
    # an existing repeatable-read snapshot after a successful owner CAS.
    if context.dialect.name == "mysql":
        from sqlalchemy.dialects.mysql import insert
        # The queue can select from this same table. Isolate that source so
        # the duplicate-key expression unambiguously names the target row.
        selection = selection.subquery("native_candidate_source").select()
        statement = insert(edges).from_select(("edge_kind", "source_id", "target_id", "revision"), selection)
        statement = statement.on_duplicate_key_update(revision=edges.c.revision + 1)
    else:
        if context.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        statement = insert(edges).from_select(("edge_kind", "source_id", "target_id", "revision"), selection)
        statement = statement.on_conflict_do_update(
            index_elements=("edge_kind", "source_id", "target_id"), set_={"revision": edges.c.revision + 1},
        )
    result = await session.execute(statement)
    if result.rowcount:
        session.info[_WORK] = True


async def enqueue(context: "SqlStorageContext", session: "AsyncSession", kind: str, target: int) -> None:
    from sqlalchemy import literal, select, true
    await _enqueue_select(context, session, select(
        literal("candidate_" + kind), literal(_EMPTY), literal(target), literal(0),
    ).where(true()))


async def replace_edges(
    context: "SqlStorageContext", session: "AsyncSession", kind: str, source: int, targets: Sequence[int],
) -> None:
    from sqlalchemy import literal, select
    await replace_owner_edges(context, session, kind, select(literal(source).label("id")), targets)


async def replace_owner_edges(
    context: "SqlStorageContext", session: "AsyncSession", kind: str, sources: "Select", targets: Sequence[int],
) -> None:
    from sqlalchemy import delete, literal, select, true, union_all, update
    edges, sequences, messages = _tables()
    target_kind = "message" if kind == "message" else "sequence"
    table = messages if target_kind == "message" else sequences
    targets = tuple(sorted(set(targets)))
    session.info.setdefault(_CLAIMED, set()).update((target_kind, target) for target in targets)
    for offset in range(0, len(targets), 128):
        batch = targets[offset:offset + 128]
        result = await session.execute(update(table).where(table.c.id.in_(batch)).values(
            gc_revision=table.c.gc_revision + 1, updated_at=table.c.updated_at,
        ))
        if result.rowcount != len(batch):
            raise NativeContentUnavailable()
    retired = (edges.c.edge_kind == kind, edges.c.source_id.in_(sources))
    if targets:
        retired += (edges.c.target_id.not_in(targets),)
    await _enqueue_select(context, session, select(
        literal("candidate_" + target_kind), literal(_EMPTY), edges.c.target_id, literal(0),
    ).where(*retired))
    await session.execute(delete(edges).where(*retired))
    owner_rows = sources.subquery("native_source_owners")
    for offset in range(0, len(targets), 128):
        values = [select(literal(target).label("target_id")) for target in targets[offset:offset + 128]]
        target_rows = (values[0] if len(values) == 1 else union_all(*values)).subquery("native_targets")
        selected = select(literal(kind), owner_rows.c.id, target_rows.c.target_id, literal(0)).select_from(
            owner_rows.join(target_rows, true()),
        ).where(true())
        # Source IDs come from the actual winning rows as a current DML read,
        # never an earlier repeatable-read snapshot or a second owner identity.
        if context.dialect.name == "mysql":
            from sqlalchemy.dialects.mysql import insert
            statement = insert(edges).from_select(("edge_kind", "source_id", "target_id", "revision"), selected)
            statement = statement.on_duplicate_key_update(revision=edges.c.revision)
        else:
            if context.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert
            else:
                from sqlalchemy.dialects.sqlite import insert
            statement = insert(edges).from_select(("edge_kind", "source_id", "target_id", "revision"), selected)
            statement = statement.on_conflict_do_nothing(index_elements=("edge_kind", "source_id", "target_id"))
        await session.execute(statement)


async def resolve_targets(
    session: "AsyncSession", kind: str, store: str, identities: Sequence[str], *, required: bool = True,
) -> dict[str, int]:
    from sqlalchemy import select
    _, sequences, messages = _tables()
    table = messages if kind == "message" else sequences
    result = {}
    identities = tuple(sorted(set(identities)))
    for offset in range(0, len(identities), 128):
        rows = (await session.execute(select(table.c.sha256, table.c.id).where(
            table.c.store_digest == store, table.c.sha256.in_(identities[offset:offset + 128]),
        ))).all()
        result.update(rows)
    if required and len(result) != len(identities):
        raise NativeContentUnavailable()
    return result


async def index_nodes(
    context: "SqlStorageContext", session: "AsyncSession", store: str, rows: Sequence[Mapping[str, object]], *,
    partial_restore: bool = False,
) -> dict[str, int]:
    from sqlalchemy import update
    if not rows:
        return {}
    edges, node_table, message_table = _tables()
    sequences = await resolve_targets(session, "sequence", store, tuple(
        identity for row in rows for identity in (row["sha256"], row["left_sha256"], row["right_sha256"], row["next_sha256"])
        if identity is not None
    ), required=not partial_restore)
    messages = await resolve_targets(session, "message", store, tuple(row["message_sha256"] for row in rows),
                                     required=not partial_restore)
    values = []
    for row in rows:
        source = sequences[row["sha256"]]
        for target in {sequences[row[name]] for name in ("left_sha256", "right_sha256", "next_sha256")
                       if row[name] in sequences}:
            values.append({"edge_kind": "sequence", "source_id": source, "target_id": target, "revision": 0})
        if row["message_sha256"] in messages:
            values.append({"edge_kind": "message", "source_id": source,
                           "target_id": messages[row["message_sha256"]], "revision": 0})
    # Immutable nodes never replace their outgoing edges. Only partial portable
    # closure imports can omit dependencies outside their retained ranges.
    for kind, table in (("sequence", node_table), ("message", message_table)):
        targets = sorted({row["target_id"] for row in values if row["edge_kind"] == kind})
        session.info.setdefault(_CLAIMED, set()).update((kind, target) for target in targets)
        for offset in range(0, len(targets), 128):
            batch = targets[offset:offset + 128]
            result = await session.execute(update(table).where(table.c.id.in_(batch)).values(
                gc_revision=table.c.gc_revision + 1, updated_at=table.c.updated_at,
            ))
            if result.rowcount != len(batch):
                raise NativeContentUnavailable()
    for offset in range(0, len(values), 128):
        await context.dialect.insert_ignore_conflict_many(session, table=edges, rows=values[offset:offset + 128],
            index_elements=("edge_kind", "source_id", "target_id"))
    return sequences


async def note_publication(
    session: "AsyncSession", store: str, roots: Sequence[str], *, resolved: Mapping[str, int] | None = None,
) -> dict[str, int]:
    # Reuse only IDs read while indexing this publication's same-store nodes.
    known = {} if resolved is None else resolved
    targets = {root: known[root] for root in sorted(set(roots)) if root in known}
    targets.update(await resolve_targets(session, "sequence", store, tuple(root for root in roots if root not in targets)))
    session.info.setdefault(_PUBLISHED, set()).update(targets.values())
    return targets


async def finish_publications(context: "SqlStorageContext", session: "AsyncSession") -> None:
    session.info.pop(_PUBLICATION_HANDOFF, None)
    claimed = session.info.get(_CLAIMED, set())
    for target in session.info.get(_PUBLISHED, ()):
        if ("sequence", target) not in claimed:
            await enqueue(context, session, "sequence", target)


async def flush_owners(context: "SqlStorageContext", session: "AsyncSession", *, group_id: str | None) -> None:
    import hashlib
    import json
    from sqlalchemy import JSON, column, select, table
    pending = session.info.pop(_PENDING, {})
    if group_id is None and any(pending.values()):
        records = table("ai_state_records", column("store_digest"), column("key_digest"), column("payload_json", JSON()))
        wire = await session.scalar(select(records.c.payload_json).where(
            records.c.store_digest == hashlib.sha256(b"linktools-ai:state-group-metadata").hexdigest(),
            records.c.key_digest == hashlib.sha256(b"archive_scope_id").hexdigest(),
        ))
        if wire is not None:
            group_id = json.loads(wire)["archive_scope_id"]
    for (kind, store, identity), roots in pending.items():
        names = {"record": "ai_state_records", "fact": "ai_state_facts", "operation": "ai_state_operations"}
        key_column = "operation_key" if kind == "operation" else "key_digest"
        owner = table(names[kind], column("id"), column("store_digest"), column(key_column),
                      column("stream_digest"), column("sequence"))
        conditions = (owner.c.store_digest == store,)
        if kind == "fact":
            conditions += (owner.c.stream_digest == identity[0], owner.c.sequence == identity[1])
        else:
            conditions += (owner.c[key_column] == identity[0],)
        sources = select(owner.c.id).where(*conditions)
        targets = []
        for root in roots:
            if group_id is None or root.group_id != group_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            targets.extend(await _owner_targets(session, root.store_digest, tuple(span.first for span in root.slices)))
        await replace_owner_edges(context, session, kind, sources, targets)


async def collect_native_content(
    context: "SqlStorageContext", session: "AsyncSession", *, store_digests: Sequence[str], budget: int = 32,
) -> None:
    from sqlalchemy import and_, delete, exists, or_, select
    edges, sequences, messages = _tables()
    if not store_digests:
        return
    scoped = or_(
        and_(edges.c.edge_kind == "candidate_sequence", or_(
            exists(select(sequences.c.id).where(
                sequences.c.id == edges.c.target_id, sequences.c.store_digest.in_(tuple(store_digests)),
            )), ~exists(select(sequences.c.id).where(sequences.c.id == edges.c.target_id)),
        )),
        and_(edges.c.edge_kind == "candidate_message", or_(
            exists(select(messages.c.id).where(
                messages.c.id == edges.c.target_id, messages.c.store_digest.in_(tuple(store_digests)),
            )), ~exists(select(messages.c.id).where(messages.c.id == edges.c.target_id)),
        )),
    )
    remaining = budget
    while remaining > 0:
        candidates = (await session.execute(select(edges).where(
            edges.c.edge_kind.in_(("candidate_sequence", "candidate_message")), scoped,
        ).order_by(edges.c.edge_kind, edges.c.source_id, edges.c.target_id).limit(remaining))).mappings().all()
        if not candidates:
            return
        remaining -= len(candidates)
        for candidate in candidates:
            is_message = candidate["edge_kind"] == "candidate_message"
            table = messages if is_message else sequences
            row = (await session.execute(select(table.c.id, table.c.store_digest, table.c.sha256, table.c.gc_revision).where(
                table.c.id == candidate["target_id"],
            ))).mappings().one_or_none()
            if row is not None:
                inbound = exists(select(edges.c.id).where(
                    edges.c.target_id == row["id"],
                    edges.c.edge_kind.in_(_MESSAGE_EDGES if is_message else _SEQUENCE_EDGES),
                ))
                removed = await session.execute(delete(table).where(
                    table.c.id == row["id"], table.c.gc_revision == row["gc_revision"],
                    table.c.store_digest.in_(tuple(store_digests)), ~inbound,
                ))
                if removed.rowcount == 1 and not is_message:
                    source = row["id"]
                    await replace_edges(context, session, "sequence", source, ())
                    await replace_edges(context, session, "message", source, ())
            await session.execute(delete(edges).where(
                edges.c.id == candidate["id"], edges.c.revision == candidate["revision"],
            ))



__all__ = ["NativeContentUnavailable", "PublicationTargetsUnavailable", "stage_owner", "register_recovery", "register_publication_targets", "run_native_mutation",
           "maintain_native_content", "replace_edges", "replace_owner_edges", "resolve_targets", "index_nodes", "note_publication", "finish_publications",
           "flush_owners", "collect_native_content", "enqueue"]
