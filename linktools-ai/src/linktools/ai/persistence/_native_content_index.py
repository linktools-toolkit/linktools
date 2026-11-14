#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transaction-scoped validation of immutable Native publication sources."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, TypeVar

from ..errors import AIError, ErrorCode
from ..steps import native_store_locator
from ._step_schema import build_step_sql_metadata

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from ..storage import SqlStorageContext


ValueT = TypeVar("ValueT")
_PENDING = "linktools.native_owner_changes"
_RECOVER = "linktools.native_recovery"
_PUBLICATION_HANDOFF = "linktools.native_publication_handoff"


class PublicationUnavailable(ValueError):
    """A receipt cannot confirm roots in the requested publication scope."""


class _PublicationConfirmation(Protocol):
    def __call__(
        self, session: "AsyncSession", *, store_key: bytes, roots: Sequence[str],
    ) -> None: ...


def register_publication(
    session: "AsyncSession", store: bytes, roots: Sequence[str], resolve: _PublicationConfirmation,
) -> None:
    """Hand one receipt's live evidence to its pending durable owner."""
    resolve(session, store_key=store, roots=roots)
    transaction = session.get_nested_transaction() or session.get_transaction()
    current = session.info.get(_PUBLICATION_HANDOFF)
    if current is None or current[0] is not transaction:
        current = (transaction, {})
        session.info[_PUBLICATION_HANDOFF] = current
    current[1][store, tuple(sorted(set(roots)))] = resolve


class NativeContentUnavailable(Exception):
    """A dependency disappeared before its new owner could commit."""

    def __init__(self, recoveries: Sequence[Callable[[], Awaitable[None]]] = ()) -> None:
        super().__init__("Native content publication requires fresh materialization")
        self.recoveries = tuple(recoveries)


def stage_owner(
    session: "AsyncSession", kind: str, store: bytes, identity: Sequence[str | int | bytes],
    data: object | None, *, state: object = None, inserting: bool = False,
) -> None:
    from ._archive_closure import iter_native_message_roots

    roots = () if data is None else tuple(iter_native_message_roots(data))
    session.info.setdefault(_PENDING, {})[(kind, store, tuple(identity))] = roots


def register_recovery(session: "AsyncSession", recovery: Callable[[], Awaitable[None]]) -> None:
    session.info.setdefault(_RECOVER, set()).add(recovery)


async def run_native_mutation(
    context: "SqlStorageContext", callback: Callable[["AsyncSession"], Awaitable[ValueT]], *, domain: str,
) -> ValueT:

    async def execute(session: "AsyncSession") -> ValueT:
        try:
            return await callback(session)
        except NativeContentUnavailable as error:
            raise NativeContentUnavailable(tuple(session.info.get(_RECOVER, ()))) from error
        finally:
            session.info.pop(_PUBLICATION_HANDOFF, None)
            session.info.pop(_PENDING, None)
            session.info.pop(_RECOVER, None)

    recovered: set[Callable[[], Awaitable[None]]] = set()
    while True:
        try:
            return await context.run_mutation(execute, domain=domain)
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


@lru_cache(maxsize=1)
def _tables():
    metadata = build_step_sql_metadata()
    return metadata.tables["ai_step_sequences"], metadata.tables["ai_step_messages"]


async def resolve_targets(
    session: "AsyncSession", kind: str, store: bytes, identities: Sequence[str], *, required: bool = True,
) -> dict[str, int]:
    from sqlalchemy import select
    sequences, messages = _tables()
    table = messages if kind == "message" else sequences
    result = {}
    identities = tuple(sorted(set(identities)))
    locator = native_store_locator(store)
    for offset in range(0, len(identities), 128):
        rows = (await session.execute(select(table.c.sha256, table.c.id).where(
            table.c.store_locator == locator, table.c.store_key == store, table.c.sha256.in_(identities[offset:offset + 128]),
        ))).all()
        result.update(rows)
    if required and len(result) != len(identities):
        raise NativeContentUnavailable()
    return result


async def index_nodes(
    context: "SqlStorageContext", session: "AsyncSession", store: bytes, rows: Sequence[Mapping[str, object]], *,
    partial_restore: bool = False,
) -> dict[str, int]:
    if not rows:
        return {}
    from sqlalchemy import literal, select, union_all

    sequence_table, message_table = _tables()
    sequence_ids = tuple(sorted({
        identity for row in rows for identity in (row["sha256"], row["left_sha256"], row["right_sha256"], row["next_sha256"])
        if identity is not None
    }))
    message_ids = tuple(sorted({row["message_sha256"] for row in rows}))
    locator = native_store_locator(store)
    sequences: dict[str, int] = {}
    messages: dict[str, int] = {}
    for offset in range(0, max(len(sequence_ids), len(message_ids)), 128):
        statements = []
        for kind, table, identities in (("sequence", sequence_table, sequence_ids),
                                         ("message", message_table, message_ids)):
            batch = identities[offset:offset + 128]
            if batch:
                statements.append(select(literal(kind), table.c.sha256, table.c.id).where(
                    table.c.store_locator == locator, table.c.store_key == store,
                    table.c.sha256.in_(batch),
                ))
        statement = statements[0] if len(statements) == 1 else union_all(*statements)
        for kind, identity, target in (await session.execute(statement)).all():
            (sequences if kind == "sequence" else messages)[identity] = target
    if not partial_restore and (len(sequences) != len(sequence_ids) or len(messages) != len(message_ids)):
        raise NativeContentUnavailable()
    # Portable slices may omit unretained descendants, never their own nodes.
    if any(row["sha256"] not in sequences for row in rows):
        raise NativeContentUnavailable()
    return sequences


async def flush_owners(context: "SqlStorageContext", session: "AsyncSession", *, group_id: str | None) -> None:
    import hashlib
    import json
    from sqlalchemy import JSON, LargeBinary, String, column, select, table
    from .state import RECORD_KEY_MAX_BYTES, STORE_KEY_MAX_BYTES, STATE_GROUP_METADATA_KEY, record_key
    pending = session.info.pop(_PENDING, {})
    if group_id is None and any(pending.values()):
        records = table("ai_state_records", column("store_key", LargeBinary(STORE_KEY_MAX_BYTES)), column("key", LargeBinary(RECORD_KEY_MAX_BYTES)), column("payload_json", JSON()), column("key_locator", String(64)))
        scope_key = record_key("", "", "archive", "archive_scope", ("archive_scope_id",))
        wire = await session.scalar(select(records.c.payload_json).where(
            records.c.store_key == STATE_GROUP_METADATA_KEY,
            records.c.key == scope_key, records.c.key_locator == hashlib.sha256(scope_key).hexdigest(),
        ))
        if wire is not None:
            group_id = json.loads(wire)["archive_scope_id"]
    targets: dict[bytes, set[str]] = {}
    for roots in pending.values():
        for root in roots:
            if group_id is None or root.group_id != group_id:
                raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
            targets.setdefault(root.store_key, set()).update(span.first for span in root.slices)
    # All owner writes have finished. Reuse live publication evidence, then
    # check overlapping dependencies once per store in this write-free phase.
    confirmed: dict[bytes, set[str]] = {}
    current = session.info.get(_PUBLICATION_HANDOFF)
    if current is not None:
        transaction = session.get_nested_transaction() or session.get_transaction()
        for (store, roots), resolve in tuple(current[1].items()):
            required = targets.get(store)
            if required is None or not required.intersection(roots):
                continue
            del current[1][store, roots]
            if current[0] is not transaction:
                continue
            try:
                # Verify the receipt's original scope, including its savepoint;
                # a subset owner's roots do not change that scope.
                resolve(session, store_key=store, roots=roots)
            except PublicationUnavailable:
                continue
            confirmed.setdefault(store, set()).update(roots)
    for store, roots in targets.items():
        missing = roots - confirmed.get(store, set())
        if missing:
            await resolve_targets(session, "sequence", store, tuple(sorted(missing)))


__all__ = ["NativeContentUnavailable", "PublicationUnavailable", "stage_owner", "register_recovery",
           "register_publication", "run_native_mutation", "resolve_targets", "index_nodes",
           "flush_owners"]
