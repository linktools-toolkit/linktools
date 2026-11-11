#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared canonical messages and bounded native occurrence reads.

This component does not provision schemas, publish owner roots, or register GC
roots. Its caller must retain roots in the actual Snapshot/Trace/Conversation
owner and include those roots and payload ObjectRefs in maintenance/export.
A native adapter supplies its existing raw store key and physical SQL context;
there is no alternative namespace/tenant identity here.
"""

import hashlib
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary, WeakSet

from ..core import canonical_json_bytes
from ..steps import native_store_locator, validate_native_store_key
from ..model.attachments import read_attachment_payload, validate_attachment_payload
from ..errors import AIError
from ..storage import ObjectRef, SqlStorageContext, StoredPayload
from ._archive import ContentRecord, ImmutableSequenceNode, PreparedArchive
from ._message_root import NativeMessageRoot, NativeMessageSequence, NativeMessageSlice
from ._payload import decode_payload_bytes
from ._step_schema import build_step_sql_metadata
from ._transcript_staging import MessageSnapshot
from .state import SqlStateTransaction, StateTransaction

if TYPE_CHECKING:
    from sqlalchemy import Table
    from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction
    from sqlalchemy.sql import Select


_MAX_COUNT = (1 << 63) - 1


def _digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("native message digest is invalid")
    try:
        if bytes.fromhex(value).hex() != value:
            raise ValueError("native message digest is invalid")
    except ValueError as error:
        raise ValueError("native message digest is invalid") from error
    return value


def _count(value: int) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError("native message count is invalid")
    return value


@dataclass(frozen=True, slots=True, init=False)
class PreparedMessagePayload:
    """An existing payload representation prepared entirely outside SQL.

    The factory must finish any ObjectStore.put before returning this value.
    ``encoded`` is that exact stored object/inline body; ``raw`` is an unchanged
    canonical singleton from MessageSnapshot. Media references remain part of
    that codec; native SDK JSON/media rewriting is not an equivalent encoder.
    """

    identity: str
    raw_size: int
    content: ContentRecord

    def __init__(self, raw: bytes, *, encoded: bytes, compression: str, payload: StoredPayload | None) -> None:
        if type(raw) is not bytes or type(encoded) is not bytes:
            raise TypeError("native messages require frozen bytes")
        identity = hashlib.sha256(raw).hexdigest()
        if payload is not None and payload.encoding == "json-attachments":
            validate_attachment_payload(raw, payload)
            if compression != "raw" or encoded != raw:
                raise ValueError("attachment payload requires the original message bytes")
        else:
            if decode_payload_bytes(encoded, codec=compression, raw_digest=identity, raw_size=len(raw)) != raw:
                raise ValueError("prepared native message payload does not match canonical bytes")
            if payload is not None:
                raise ValueError("ordinary native message bytes must remain in the database")
        content = ContentRecord("model-message", identity, len(raw), compression, payload,
                                encoded if payload is None else None)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "raw_size", len(raw))
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PreparedNativeMessages:
    root: NativeMessageSequence
    frontier: tuple[ImmutableSequenceNode, ...]
    messages: tuple[tuple[str, ContentRecord], ...]
    nodes: tuple[tuple[str, ImmutableSequenceNode], ...]
    generation: int
    cold: bool = False


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class NativeMessageReceipt:
    """Provisional content dependencies, never proof of an owner's CAS outcome.

    The insertion guard prevents a rejected candidate from gaining a receipt.
    Its root is usable in the same transaction, not for later append reuse until
    the owning mutation confirms successful commit.
    """

    prepared: PreparedNativeMessages | None
    _transaction: "AsyncSessionTransaction | None" = None
    _store_key: bytes | None = field(default=None, repr=False)
    _root_targets: tuple[tuple[str, int], ...] = ()

    def publication_targets(
        self, session: "AsyncSession", *, store_key: bytes, roots: Sequence[str],
    ) -> tuple[int, ...]:
        """Reuse physical IDs only inside this exact live publication transaction."""
        from ._native_content_index import PublicationTargetsUnavailable

        if (self.prepared is None or self._transaction is None
                or (session.get_nested_transaction() or session.get_transaction()) is not self._transaction or not self._transaction.is_active
                or store_key != self._store_key
                or tuple(sorted(set(roots))) != tuple(root for root, _ in self._root_targets)):
            raise PublicationTargetsUnavailable("native publication targets require their active transaction, store, and roots")
        return tuple(target for _, target in self._root_targets)

    @property
    def root(self) -> NativeMessageSequence | None:
        return None if self.prepared is None else self.prepared.root


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class CommittedNativeMessages:
    """Committed content closure only; no snapshot/conversation ownership claim."""

    root: NativeMessageSequence
    frontier: tuple[ImmutableSequenceNode, ...]
    generation: int


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class NativeMessageReadback:
    """Validated bytes/frontier from the owner's existing scoped body query.

    This is not reusable until that owning read operation returns successfully.
    Keep payloads only for the consumer's read lifetime; adopted proofs retain
    the sequence descriptors and logarithmic append frontier, not the bodies.
    """

    root: NativeMessageSequence
    payloads: tuple[bytes, ...]
    frontier: tuple[ImmutableSequenceNode, ...]
    generation: int


@dataclass(frozen=True, slots=True)
class NativeRangeBudget:
    max_seeds: int = 446
    max_bind_parameters: int = 900
    max_sql_bytes: int = 256 * 1024
    max_mysql_depth: int = 60
    max_path_units: int = 1_000_000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (
            self.max_seeds, self.max_bind_parameters, self.max_sql_bytes,
            self.max_mysql_depth, self.max_path_units,
        )):
            raise ValueError("native range budgets must be positive integers")


@dataclass(frozen=True, slots=True)
class NativeRangeQuery:
    statement: "Select"
    bind_parameters: int
    sql_bytes: int
    depth: int
    path_units: int


class NativeMessageContent:
    """One native content owner, borrowing the existing adapter's SQL scope.

    The native adapter supplies its public storage_context/store_key.
    Standalone snapshot sequences use that row's existing scope. Cross-owner
    references require bind_group with the actual durable group ID; publication
    additionally checks the physical SQL context. No synthetic group is minted.
    No inference from mutable SDK identity or checkpoint length is permitted.
    Immutable content identities permit byte-exact duplicate reuse without a
    separate SELECT or body rewrite. The canonical schema is required: no additional triggers,
    constraints or narrower column definitions. Owner replay keys have different
    semantics and never establish content proof on a conflicting insertion.
    """

    def __init__(
        self, context: SqlStorageContext, *, store_key: bytes, group_id: str | None = None,
        max_cached_messages: int = 256, max_cached_bytes: int = 1024 * 1024,
        max_cached_nodes: int = 1024,
    ) -> None:
        self._empty = NativeMessageSequence()
        self._store_key = validate_native_store_key(store_key)
        self._store_locator = native_store_locator(self._store_key)
        self._group_id: str | None = None
        if group_id is not None:
            self.bind_group(group_id)
        if any(type(value) is not int or value < 0 for value in (
            max_cached_messages, max_cached_bytes, max_cached_nodes,
        )):
            raise ValueError("native content cache limits must be nonnegative")
        self._context = context
        metadata = build_step_sql_metadata()
        self._messages = metadata.tables["ai_step_messages"]
        self._nodes = metadata.tables["ai_step_sequences"]
        self._generation = 0
        self._prepared: WeakSet[PreparedNativeMessages] = WeakSet()
        self._replacements: WeakKeyDictionary[PreparedNativeMessages, PreparedNativeMessages] = WeakKeyDictionary()
        self._recoveries: WeakKeyDictionary[PreparedNativeMessages, Callable[[], Awaitable[None]]] = WeakKeyDictionary()
        self._receipts: WeakSet[NativeMessageReceipt] = WeakSet()
        self._proofs: WeakSet[CommittedNativeMessages] = WeakSet()
        self._readbacks: WeakSet[NativeMessageReadback] = WeakSet()
        self._known_messages: OrderedDict[str, int] = OrderedDict()
        self._known_nodes: OrderedDict[str, None] = OrderedDict()
        self._cache_bytes = 0
        self._max_messages, self._max_bytes, self._max_nodes = (
            max_cached_messages, max_cached_bytes, max_cached_nodes,
        )

    def invalidate(self) -> None:
        """Invalidate process-local knowledge when quiesced content is reclaimed."""
        self._generation += 1
        self._prepared.clear()
        self._receipts.clear()
        self._proofs.clear()
        self._readbacks.clear()
        self._known_messages.clear()
        self._known_nodes.clear()
        self._cache_bytes = 0

    def bind_group(self, group_id: str) -> None:
        """Bind the already resolved physical group's durable scope, without I/O.

        Standalone snapshots do not resolve a group UUID. Cross-owner references
        must bind the existing group's scope before wrapping an internal value.
        """
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("native message group is required")
        if self._group_id is not None and self._group_id != group_id:
            raise ValueError("native message content is already bound to another group")
        self._group_id = group_id

    def empty_root(self) -> NativeMessageSequence:
        return self._empty

    def scoped_root(self, sequence: NativeMessageSequence) -> NativeMessageRoot:
        """Wrap a trusted internal sequence, not mint publication/ownership proof."""
        if not isinstance(sequence, NativeMessageSequence):
            raise TypeError("native message reference requires an internal sequence")
        if self._group_id is None:
            raise ValueError("native message group must be bound before cross-owner reference")
        return NativeMessageRoot(self._group_id, self._store_key, sequence.slices)

    def validate_root(self, root: NativeMessageSequence | NativeMessageRoot) -> None:
        """Check the physical owner before interpreting a cross-owner sequence."""
        self._scope(root)

    def _scope(self, root: NativeMessageSequence | NativeMessageRoot) -> None:
        if isinstance(root, NativeMessageSequence):
            return
        if (not isinstance(root, NativeMessageRoot) or self._group_id is None
                or (root.group_id, root.store_key) != (self._group_id, self._store_key)):
            raise ValueError("native message root belongs to another persistence scope")

    def _proof(self, proof: CommittedNativeMessages) -> None:
        if proof not in self._proofs or proof.generation != self._generation:
            raise ValueError("native message reuse requires this owner's committed proof")
        self._scope(proof.root)

    def select(self, proof: CommittedNativeMessages, start: int, end: int) -> CommittedNativeMessages:
        self._proof(proof)
        root = proof.root.slice(start, end)
        frontier = proof.frontier if root.slices and root.slices[-1] == proof.root.slices[-1] else ()
        result = CommittedNativeMessages(root, frontier, self._generation)
        self._proofs.add(result)
        return result

    async def prepare_append(
        self, delta: MessageSnapshot, *,
        prepare_payload: Callable[[bytes], Awaitable[PreparedMessagePayload]],
        base: CommittedNativeMessages | None = None,
        archive: PreparedArchive | None = None,
        reuse_payload: Callable[[bytes, ContentRecord], Awaitable[PreparedMessagePayload]] | None = None,
        use_cache: bool = True,
    ) -> PreparedNativeMessages:
        """Prepare O(delta) bodies/nodes plus O(log n + r) frontier/descriptors.

        Mutable inputs must first pass MessageArchive.snapshot, which encodes and
        checks all n messages. Supplying a frozen delta is an explicit append;
        neither a common prefix nor unmodified SDK objects are guessed here.
        """
        if not isinstance(delta, MessageSnapshot):
            raise TypeError("native append requires an explicitly frozen MessageSnapshot delta")
        if base is not None:
            self._proof(base)
        root = self._empty if base is None else base.root
        slices = list(root.slices)
        frontier = deque(() if base is None else base.frontier)
        tail = slices[-1] if slices else None
        if tail is None or tail.start != 0 or tail.end != tail.count or not frontier:
            frontier.clear()
            count = 0
        else:
            if (tuple(node.tree_count for node in frontier) != ImmutableSequenceNode.forest_weights(tail.count)
                    or frontier[0].identity != tail.first
                    or any(node.next_forest != (frontier[index + 1].identity if index + 1 < len(frontier) else None)
                           for index, node in enumerate(frontier))):
                raise ValueError("committed native append frontier is invalid")
            count = tail.count
            if len(delta):
                slices.pop()
        candidates: dict[str, ContentRecord] = {}
        if archive is not None and reuse_payload is not None:
            generation, candidates = archive.native_payloads.get(self, (self._generation, {}))
            if generation != self._generation:
                candidates = {}
        additions: dict[str, ContentRecord] = {}
        messages: dict[str, ContentRecord] = {}
        nodes: dict[str, ImmutableSequenceNode] = {}
        for raw in delta.encoded_payloads():
            identity = hashlib.sha256(raw).hexdigest()
            if identity not in messages and (not use_cache or identity not in self._known_messages):
                candidate = candidates.get(identity)
                if candidate is not None and (candidate.codec != "model-message"
                        or candidate.raw_digest != identity or candidate.raw_size != len(raw)):
                    raise ValueError("prepared native message identity or size is corrupt")
                payload = (await reuse_payload(raw, candidate)
                           if candidate is not None and reuse_payload is not None else await prepare_payload(raw))
                if not isinstance(payload, PreparedMessagePayload) or (payload.identity, payload.raw_size) != (identity, len(raw)):
                    raise ValueError("native message payload factory changed canonical content")
                messages[identity] = payload.content
                if archive is not None and reuse_payload is not None:
                    additions[identity] = payload.content
            first = frontier[0] if frontier else None
            second = frontier[1] if len(frontier) > 1 else None
            node = ImmutableSequenceNode.extend(identity, first, second)
            if first is not None and second is not None and first.tree_count == second.tree_count:
                frontier.popleft()
                frontier.popleft()
            frontier.appendleft(node)
            if not use_cache or node.identity not in self._known_nodes:
                nodes[node.identity] = node
            count = _count(count + 1)
        if len(delta):
            slices.append(NativeMessageSlice(frontier[0].identity, count, 0, count))
        prepared = PreparedNativeMessages(
            NativeMessageSequence(tuple(slices)), tuple(frontier),
            tuple(messages.items()), tuple(nodes.items()), self._generation, not use_cache,
        )
        self._prepared.add(prepared)
        if archive is not None and reuse_payload is not None:
            candidates.update(additions)
            archive.native_payloads[self] = self._generation, candidates
        return prepared

    def bind_recovery(
        self, prepared: PreparedNativeMessages, recover: Callable[[], Awaitable[None]],
    ) -> None:
        self._recoveries[prepared] = recover

    def replace_prepared(self, previous: PreparedNativeMessages, replacement: PreparedNativeMessages) -> None:
        if previous.root != replacement.root or replacement not in self._prepared:
            raise ValueError("Cold native materialization changed the owner's root")
        self._replacements[previous] = replacement

    async def publish(
        self, transaction: StateTransaction, prepared: PreparedNativeMessages, *,
        owner_inserted: bool, confirmed_content: CommittedNativeMessages | None = None,
    ) -> NativeMessageReceipt:
        """Publish after an owner's actual INSERT result in its group transaction.

        A rejected keyed candidate always receives an empty receipt. Existing
        content proof cannot identify the root held by a conflicting owner row.
        Receipts only prove content dependencies, never an owner's CAS outcome.
        """
        return await self.publish_retained(transaction, prepared, owner_written=owner_inserted,
                                           confirmed_content=confirmed_content)

    async def publish_retained(
        self, transaction: StateTransaction, prepared: PreparedNativeMessages, *,
        owner_written: bool, confirmed_content: CommittedNativeMessages | None = None,
    ) -> NativeMessageReceipt:
        """Publish after a real owner INSERT or successful CAS retaining this root.

        ``owner_written`` is the completed write's outcome, not intent or merely
        existing content. The caller must let every failure abort this same
        transaction. Confirm only after the outer mutation returns successfully.
        """
        if not isinstance(transaction, SqlStateTransaction):
            raise TypeError("native messages require a SQL state transaction")
        return await transaction.run_sql(self._context, lambda session: self._publish_in_session(
            session, prepared, owner_written=owner_written, confirmed_content=confirmed_content,
        ))

    async def publish_in_session(
        self, session: "AsyncSession", prepared: PreparedNativeMessages, *,
        owner_inserted: bool, confirmed_content: CommittedNativeMessages | None = None,
    ) -> NativeMessageReceipt:
        """Join the native adapter's already active, same-engine INSERT transaction."""
        return await self._publish_in_session(session, prepared, owner_written=owner_inserted,
                                              confirmed_content=confirmed_content)

    async def _publish_in_session(
        self, session: "AsyncSession", prepared: PreparedNativeMessages, *,
        owner_written: bool, confirmed_content: CommittedNativeMessages | None,
    ) -> NativeMessageReceipt:
        from sqlalchemy.ext.asyncio import AsyncConnection
        from ._native_content_index import index_nodes, note_publication, register_publication_targets, register_recovery

        recovery = self._recoveries.get(prepared)
        if recovery is not None:
            register_recovery(session, recovery)
        prepared = self._replacements.get(prepared, prepared)
        if prepared not in self._prepared or prepared.generation != self._generation:
            raise ValueError("native publication requires this owner's prepared content")
        if type(owner_written) is not bool:
            raise TypeError("owner write outcome must be explicit")
        if confirmed_content is not None:
            self._proof(confirmed_content)
            if confirmed_content.root != prepared.root:
                raise ValueError("committed native content proof does not match candidate root")
        bind = session.bind
        engine = bind.engine if isinstance(bind, AsyncConnection) else bind
        if engine is not self._context.engine or self._context.closed or not session.in_transaction():
            raise ValueError("native publication requires an active same-engine SQL transaction")
        if not owner_written:
            receipt = NativeMessageReceipt(None)
        else:
            if confirmed_content is None or prepared.cold:
                await self._insert_rows(session, self._messages, [
                    {"store_key": self._store_key, "store_locator": self._store_locator, "sha256": identity, **body.to_native_values()}
                    for identity, body in prepared.messages if prepared.cold or identity not in self._known_messages
                ])
                await self._insert_rows(session, self._nodes, [
                    {"store_key": self._store_key, "store_locator": self._store_locator, "sha256": identity,
                     "message_sha256": node.content_id, "tree_count": node.tree_count,
                     "left_sha256": node.left, "right_sha256": node.right, "next_sha256": node.next_forest}
                    for identity, node in prepared.nodes if prepared.cold or identity not in self._known_nodes
                ])
            resolved = await index_nodes(self._context, session, self._store_key, [
                {"sha256": identity, "message_sha256": node.content_id,
                 "left_sha256": node.left, "right_sha256": node.right, "next_sha256": node.next_forest}
                for identity, node in prepared.nodes
            ])
            targets = await note_publication(
                session, self._store_key, tuple(span.first for span in prepared.root.slices), resolved=resolved,
            )
            receipt = NativeMessageReceipt(
                prepared, session.get_nested_transaction() or session.get_transaction(),
                self._store_key, tuple(sorted(targets.items())),
            )
            register_publication_targets(
                session, self._store_key, tuple(span.first for span in prepared.root.slices),
                receipt.publication_targets,
            )
        self._receipts.add(receipt)
        return receipt

    async def _insert_rows(self, session: "AsyncSession", table: "Table", rows: Sequence[Mapping[str, object]]) -> None:
        # Compare the winning immutable row inside INSERT. An exact duplicate
        # leaves its surrogate ID, audit columns and body unchanged. A mismatch
        # must abort the enclosing owner mutation, including earlier batches.
        batch: list[Mapping[str, object]] = []
        size = 0
        for row in rows:
            validate_native_store_key(row["store_key"])
            _digest(row["sha256"])
            if table is self._messages:
                ContentRecord.from_native_row(row)
            else:
                node = ImmutableSequenceNode(_digest(row["message_sha256"]), _count(row["tree_count"]),
                                             row["left_sha256"], row["right_sha256"], row["next_sha256"])
                if node.identity != row["sha256"]:
                    raise ValueError("native message sequence identity is invalid")
            row_size = sum(len(value.encode("utf-8")) if isinstance(value, str) else len(value)
                           for value in row.values() if isinstance(value, (str, bytes)))
            if batch and (len(batch) >= 64 or size + row_size > 256 * 1024):
                await self._context.dialect.insert_immutable_many(
                    session, table=table, rows=batch,
                    index_elements=("store_locator", "sha256"), conflict_column="id",
                )
                batch, size = [], 0
            batch.append(row)
            size += row_size
        if batch:
            await self._context.dialect.insert_immutable_many(
                session, table=table, rows=batch,
                index_elements=("store_locator", "sha256"), conflict_column="id",
            )

    def confirm_current(self, receipt: NativeMessageReceipt) -> CommittedNativeMessages | None:
        """An expired optimization receipt cannot change an already committed outcome."""
        if receipt.prepared is None or receipt.prepared.generation != self._generation:
            return None
        return self.confirm_committed(receipt)

    def discard_uncommitted(self, receipts: Sequence[NativeMessageReceipt]) -> None:
        """Revoke failed mutation receipts without invalidating frozen retry input."""
        for receipt in receipts:
            self._receipts.discard(receipt)

    def confirm_committed(self, receipt: NativeMessageReceipt) -> CommittedNativeMessages:
        """Record the owning mutation's confirmed successful return.

        Call only after StateStore.mutate/SqlStorageContext.run_mutation returns
        successfully, never from its SQL callback or after COMMIT_UNKNOWN. This
        is the existing owner confirmation boundary, not a new commit detector.
        Unknown outcomes require the owner's existing durable readback first.
        """
        if (receipt not in self._receipts or receipt.prepared is None
                or receipt.prepared.generation != self._generation):
            raise ValueError("native message receipt has no confirmed committed root")
        prepared = receipt.prepared
        for identity, body in prepared.messages:
            if identity not in self._known_messages and self._max_messages and self._max_bytes:
                size = len(body.body) if body.body is not None else len(canonical_json_bytes(body.payload.to_json()))
                self._known_messages[identity] = size
                self._cache_bytes += size
        while len(self._known_messages) > self._max_messages or self._cache_bytes > self._max_bytes:
            _, body = self._known_messages.popitem(last=False)
            self._cache_bytes -= body
        for identity, _ in prepared.nodes:
            self._known_nodes[identity] = None
        while len(self._known_nodes) > self._max_nodes:
            self._known_nodes.popitem(last=False)
        proof = CommittedNativeMessages(prepared.root, prepared.frontier, self._generation)
        self._proofs.add(proof)
        return proof

    @classmethod
    def _range_shape(cls, item: NativeMessageSlice) -> tuple[int, int]:
        """Count the intersecting paths from shape alone, without visiting leaves."""
        lo, hi = item.count - item.end, item.count - item.start
        offset = deepest = visits = 0
        for forest_depth, size in enumerate(ImmutableSequenceNode.forest_weights(item.count)):
            if offset >= hi:
                break
            if offset + size <= lo:
                # The next forest tree is reached through this head, not its children.
                visits += 1
                deepest = max(deepest, forest_depth)
            else:
                pending = [(offset, size, forest_depth)]
                while pending:
                    position, count, depth = pending.pop()
                    if position >= hi or position + count <= lo:
                        continue
                    if lo <= position and position + count <= hi:
                        visits += count
                        deepest = max(deepest, depth + count.bit_length() - 1)
                        continue
                    visits += 1
                    deepest = max(deepest, depth)
                    half = (count - 1) // 2
                    if half:
                        pending.extend(((position + 1, half, depth + 1),
                                        (position + 1 + half, half, depth + 1)))
            offset += size
        return deepest, visits

    def _range_inputs(
        self, roots: Sequence[NativeMessageSequence | NativeMessageRoot],
    ) -> tuple[tuple[tuple[int, int, NativeMessageSlice], ...], dict[tuple[int, int], tuple[int, int]]]:
        seeds: dict[NativeMessageSlice, tuple[int, int]] = {}
        aliases: dict[tuple[int, int], tuple[int, int]] = {}
        for request, root in enumerate(roots):
            self._scope(root)
            for span, item in enumerate(root.slices):
                aliases[request, span] = seeds.setdefault(item, (request, span))
        return tuple((request, span, item) for item, (request, span) in seeds.items()), aliases

    def range_query(
        self, roots: Sequence[NativeMessageSequence | NativeMessageRoot], *, budget: NativeRangeBudget = NativeRangeBudget(),
    ) -> NativeRangeQuery | None:
        """One body SELECT, traversing intersecting subtrees only.

        MySQL 5.7 expands bounded derived frontiers. Its repeated path work is
        explicitly budgeted, not equated with the recursive CTE's CPU cost.
        Over-budget requests fail before execution; callers choose pagination
        at their existing boundary rather than silently adding SELECTs.
        """
        from sqlalchemy import BigInteger, and_, bindparam, case, cast, literal_column, null, select, union_all

        dialect = self._context.engine.dialect
        integer = lambda value: cast(literal_column(str(value)), BigInteger)
        true = literal_column("1 = 1")
        nodes, messages = self._nodes, self._messages
        store = bindparam("native_store_key", self._store_key, type_=nodes.c.store_key.type)
        locator = bindparam("native_store_locator", self._store_locator, type_=nodes.c.store_locator.type)
        seeds = []
        depth = path_units = 0
        inputs, _ = self._range_inputs(roots)
        if len(inputs) > budget.max_seeds:
            raise ValueError("native range seed budget exceeded")
        if inputs and all(item.count == 1 for _, _, item in inputs):
            statement = select(
                null().label("request"), null().label("span"), integer(0).label("position"),
                nodes.c.sha256, nodes.c.message_sha256, nodes.c.tree_count,
                nodes.c.left_sha256, nodes.c.right_sha256, nodes.c.next_sha256, messages.c.message, messages.c.compression, messages.c.raw_size, messages.c.payload_json,
            ).select_from(nodes.outerjoin(messages, and_(
                messages.c.store_locator == locator, messages.c.store_key == store, messages.c.sha256 == nodes.c.message_sha256,
            ))).where(nodes.c.store_locator == locator, nodes.c.store_key == store, nodes.c.sha256.in_(
                bindparam("native_singleton_ids", [item.first for _, _, item in inputs], expanding=True),
            ))
            return self._range_plan(statement, depth=0, path_units=len(inputs), budget=budget)
        for request, span, item in inputs:
            height, visits = self._range_shape(item)
            depth = max(depth, height)
            path_units += visits
            index = len(seeds)
            root_id = bindparam(f"native_root_{index}", item.first, type_=nodes.c.sha256.type)
            if dialect.name == "mysql":
                from sqlalchemy.dialects.mysql import CHAR

                root_id = cast(root_id, CHAR(64, charset="utf8mb4")).collate("utf8mb4_bin")
            else:
                root_id = cast(root_id, nodes.c.sha256.type)
            seeds.append(select(
                integer(request).label("request"), integer(span).label("span"),
                root_id.label("sha256"),
                integer(item.count).label("remaining"), integer(0).label("position"),
                integer(item.count - item.end).label("lo"), integer(item.count - item.start).label("hi"),
                integer(0).label("depth"), integer(1).label("forest"),
            ))
        if not seeds:
            return None
        mysql = dialect.name == "mysql"
        if mysql:
            if depth > budget.max_mysql_depth:
                raise ValueError("native range MySQL depth budget exceeded")
            path_units *= depth + 1
        if path_units > budget.max_path_units:
            raise ValueError("native range path-work budget exceeded")
        start = (seeds[0] if len(seeds) == 1 else union_all(*seeds)).subquery("native_range_roots")
        edges = union_all(*(select(integer(index).label("edge")) for index in range(4 if mysql else 3))).subquery("native_range_edges")

        def extend(frontier):
            half = cast((nodes.c.tree_count - integer(1)) / integer(2), BigInteger)
            child = case((edges.c.edge == integer(0), nodes.c.left_sha256),
                         (edges.c.edge == integer(1), nodes.c.right_sha256),
                         (edges.c.edge == integer(2), nodes.c.next_sha256), else_=frontier.c.sha256)
            position = case((edges.c.edge == integer(0), frontier.c.position + integer(1)),
                            (edges.c.edge == integer(1), frontier.c.position + integer(1) + half),
                            (edges.c.edge == integer(2), frontier.c.position + nodes.c.tree_count),
                            else_=frontier.c.position)
            remaining = case((edges.c.edge == integer(2), frontier.c.remaining - nodes.c.tree_count),
                             (edges.c.edge == integer(3), frontier.c.remaining), else_=half)
            return select(
                frontier.c.request, frontier.c.span, child.label("sha256"), remaining.label("remaining"),
                position.label("position"), frontier.c.lo, frontier.c.hi,
                (frontier.c.depth + integer(1)).label("depth"),
                case((edges.c.edge == integer(2), integer(1)),
                     (edges.c.edge == integer(3), frontier.c.forest), else_=integer(0)).label("forest"),
            ).select_from(frontier.join(nodes, and_(
                nodes.c.store_locator == locator, nodes.c.store_key == store, nodes.c.sha256 == frontier.c.sha256,
            )).join(edges, true)).where(
                child.is_not(None), frontier.c.depth < integer(depth),
                nodes.c.tree_count > integer(0), nodes.c.tree_count <= frontier.c.remaining,
                (edges.c.edge < integer(2)) | (edges.c.edge == integer(3)) | (frontier.c.forest == integer(1)),
                remaining > integer(0), position < frontier.c.hi, position + remaining > frontier.c.lo,
            )

        if mysql:
            levels = [start]
            for level in range(depth):
                levels.append(extend(levels[-1]).distinct().subquery(f"native_range_level_{level + 1}"))
            # Carry prior nodes through a self edge. DISTINCT includes occurrence
            # position, so repeated content is retained while repeated paths are
            # collapsed. Selecting only the final frontier keeps SQL text linear
            # in depth; internal node/index work still grows with depth * visits.
            closure = levels[-1]
        else:
            closure = select(start).cte("native_range_nodes", recursive=True)
            closure = closure.union_all(extend(closure))
        selected = and_(closure.c.position >= closure.c.lo, closure.c.position < closure.c.hi)
        statement = select(
            closure.c.request, closure.c.span, closure.c.position, nodes.c.sha256,
            nodes.c.message_sha256, nodes.c.tree_count, nodes.c.left_sha256, nodes.c.right_sha256,
            nodes.c.next_sha256, messages.c.message, messages.c.compression, messages.c.raw_size, messages.c.payload_json,
        ).select_from(closure.join(nodes, and_(
            nodes.c.store_locator == locator, nodes.c.store_key == store, nodes.c.sha256 == closure.c.sha256,
        )).outerjoin(messages, and_(
            selected, messages.c.store_locator == locator, messages.c.store_key == store, messages.c.sha256 == nodes.c.message_sha256,
        )))
        return self._range_plan(statement, depth=depth, path_units=path_units, budget=budget)

    def _range_plan(
        self, statement: "Select", *, depth: int, path_units: int, budget: NativeRangeBudget,
    ) -> NativeRangeQuery:
        if path_units > budget.max_path_units:
            raise ValueError("native range path-work budget exceeded")
        compiled = statement.compile(dialect=self._context.engine.dialect, compile_kwargs={"render_postcompile": True})
        binds = len(compiled.positiontup) if compiled.positiontup is not None else len(compiled.params)
        sql_bytes = len(str(compiled).encode("utf-8"))
        if binds > budget.max_bind_parameters or sql_bytes > budget.max_sql_bytes:
            raise ValueError("native range SQL/bind budget exceeded")
        return NativeRangeQuery(statement, binds, sql_bytes, depth, path_units)

    async def hydrate_ranges(
        self, roots: Sequence[NativeMessageSequence | NativeMessageRoot], rows: Sequence[Mapping[str, object]], *,
        read_object: Callable[[ObjectRef], Awaitable[bytes]] | None = None,
    ) -> tuple[tuple[bytes, ...], ...]:
        """Validate traversed metadata and decode only selected canonical bodies.

        Object reads occur after the body query, outside its SQL transaction.
        The caller's existing model-message decoder restores complete SDK and
        business fields; this layer returns their exact original bytes.
        """
        roots = tuple(roots)
        inputs, aliases = self._range_inputs(roots)
        singletons = {item.first: (request, span) for request, span, item in inputs} if all(
            item.count == 1 for _, _, item in inputs
        ) else {}
        indexed: dict[tuple[int, int, int], Mapping[str, object]] = {}
        for row in rows:
            if row["request"] is None and row["span"] is None:
                source = singletons.get(row["sha256"])
                if source is None:
                    raise ValueError("native range returned an unrelated singleton")
                key = (*source, row["position"])
            else:
                key = (row["request"], row["span"], row["position"])
            if any(type(value) is not int for value in key) or key in indexed:
                raise ValueError("native range returned duplicate or invalid occurrences")
            indexed[key] = row
        bodies: dict[str, bytes] = {}
        representations: dict[str, ContentRecord] = {}
        results: list[tuple[bytes, ...]] = []
        consumed: set[tuple[int, int, int]] = set()
        for request, root in enumerate(roots):
            output: list[bytes] = []
            for span, item in enumerate(root.slices):
                lo, hi = item.count - item.end, item.count - item.start
                pending = [(item.first, item.count, 0, True)]
                selected: dict[int, bytes] = {}
                while pending:
                    identity, remaining, position, forest = pending.pop()
                    if position >= hi or position + remaining <= lo:
                        continue
                    source_request, source_span = aliases[request, span]
                    key = source_request, source_span, position
                    row = indexed.get(key)
                    if row is None:
                        raise ValueError("native range sequence is incomplete")
                    consumed.add(key)
                    try:
                        node = ImmutableSequenceNode(_digest(row["message_sha256"]), row["tree_count"],
                                                     row["left_sha256"], row["right_sha256"], row["next_sha256"])
                    except AIError as error:
                        raise ValueError("native range sequence is corrupt") from error
                    expected = ImmutableSequenceNode.forest_weights(remaining)[0] if forest else remaining
                    if row["sha256"] != identity or node.identity != identity or node.tree_count != expected:
                        raise ValueError("native range sequence is corrupt")
                    if forest and ((remaining > expected) != (node.next_forest is not None)):
                        raise ValueError("native range forest is incomplete")
                    if lo <= position < hi:
                        try:
                            content = ContentRecord.from_native_row(row)
                        except (AIError, TypeError, ValueError) as error:
                            raise ValueError("native selected message is missing or corrupt") from error
                        if content.codec != "model-message" or content.raw_digest != node.content_id:
                            raise ValueError("native message codec or identity is corrupt")
                        previous = representations.setdefault(node.content_id, content)
                        if previous != content:
                            raise ValueError("native message has conflicting stored representations")
                        if node.content_id not in bodies:
                            if content.payload is None:
                                encoded = content.body
                            elif content.payload.encoding == "json-attachments":
                                if read_object is None:
                                    raise ValueError("native message ObjectStore resolver is required")
                                encoded = await read_attachment_payload(content.payload, read_object=read_object)
                            else:
                                raise ValueError("native message payload is not an attachment manifest")
                            bodies[node.content_id] = decode_payload_bytes(
                                encoded, codec=content.compression, raw_digest=content.raw_digest, raw_size=content.raw_size,
                            )
                        if len(bodies[node.content_id]) != content.raw_size:
                            raise ValueError("native message size is corrupt")
                        selected[position] = bodies[node.content_id]
                    elif any(row[name] is not None for name in ("message", "compression", "raw_size", "payload_json")):
                        raise ValueError("native range returned an unselected message body")
                    if expected > 1:
                        half = (expected - 1) // 2
                        pending.extend(((node.left, half, position + 1, False),
                                        (node.right, half, position + 1 + half, False)))
                    if forest and remaining > expected:
                        pending.append((node.next_forest, remaining - expected, position + expected, True))
                if len(selected) != hi - lo:
                    raise ValueError("native range message occurrences are incomplete")
                output.extend(selected[position] for position in range(hi - 1, lo - 1, -1))
            results.append(tuple(output))
        if consumed != set(indexed):
            raise ValueError("native range returned unrelated sequence nodes")
        return tuple(results)

    def partition_range_rows(
        self, roots: Sequence[NativeMessageSequence | NativeMessageRoot], rows: Sequence[Mapping[str, object]],
    ) -> tuple[tuple[Mapping[str, object], ...], ...]:
        """Isolate corrupt owners from one already executed batched SELECT."""
        inputs, aliases = self._range_inputs(roots)
        singletons = {item.first: (request, span) for request, span, item in inputs}
        indexed: dict[tuple[int, int], list[Mapping[str, object]]] = {}
        valid = {(request, span) for request, span, _item in inputs}
        for row in rows:
            source = (singletons.get(row["sha256"]) if row["request"] is None and row["span"] is None
                      else (row["request"], row["span"]))
            if source not in valid:
                raise ValueError("native range returned unrelated rows")
            indexed.setdefault(source, []).append(row)
        result = []
        for request, root in enumerate(roots):
            local, _aliases = self._range_inputs((root,))
            result.append(tuple(dict(row, request=0, span=span)
                for _local_request, span, _item in local
                for row in indexed.get(aliases[request, span], ())))
        return tuple(result)

    async def read_hydrated_ranges(
        self, roots: Sequence[NativeMessageSequence | NativeMessageRoot], rows: Sequence[Mapping[str, object]], *,
        read_object: Callable[[ObjectRef], Awaitable[bytes]] | None = None,
    ) -> tuple[NativeMessageReadback, ...]:
        """Validate the actual rows from the caller's existing scoped body SELECT.

        Internal sequences must come from that same native owner's filtered
        header slot. Scoped roots additionally verify the actual group ID. This
        operation never reads SQL or grants proof from a descriptor alone. The
        caller adopts each result only after its enclosing read succeeds.
        """
        roots = tuple(roots)
        rows = tuple(dict(row) for row in rows)
        payloads = await self.hydrate_ranges(roots, rows, read_object=read_object)
        inputs, aliases = self._range_inputs(roots)
        singletons = {item.first: (request, span) for request, span, item in inputs} if all(
            item.count == 1 for _, _, item in inputs
        ) else {}
        indexed = {}
        for row in rows:
            source = (singletons[row["sha256"]] if row["request"] is None and row["span"] is None
                      else (row["request"], row["span"]))
            indexed[(*source, row["position"])] = row
        results = []
        for request, (root, values) in enumerate(zip(roots, payloads, strict=True)):
            sequence = root.sequence if isinstance(root, NativeMessageRoot) else root
            frontier = []
            tail = sequence.slices[-1] if sequence.slices else None
            if tail is not None and tail.start == 0 and tail.end == tail.count:
                source = aliases[request, len(sequence.slices) - 1]
                position = 0
                for count in ImmutableSequenceNode.forest_weights(tail.count):
                    row = indexed[(*source, position)]
                    frontier.append(ImmutableSequenceNode(_digest(row["message_sha256"]), row["tree_count"],
                                                         row["left_sha256"], row["right_sha256"], row["next_sha256"]))
                    position += count
            result = NativeMessageReadback(sequence, values, tuple(frontier), self._generation)
            self._readbacks.add(result)
            results.append(result)
        return tuple(results)

    def adopt_read(self, result: NativeMessageReadback) -> CommittedNativeMessages:
        """Adopt validated readback after the owner's enclosing read succeeds.

        A failed/unknown read is not an adoption boundary. This mirrors write
        confirmation without a second query, transaction listener or body cache.
        """
        if result not in self._readbacks or result.generation != self._generation:
            raise ValueError("native message adoption requires this owner's validated readback")
        proof = CommittedNativeMessages(result.root, result.frontier, self._generation)
        self._proofs.add(proof)
        return proof


__all__ = [
    "NativeMessageSlice", "NativeMessageRoot", "NativeMessageSequence", "PreparedMessagePayload", "PreparedNativeMessages",
    "NativeMessageReceipt", "CommittedNativeMessages", "NativeMessageReadback", "NativeRangeBudget", "NativeRangeQuery", "NativeMessageContent",
]
