#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact scoped streams share counter identity across current State backends."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.steps import native_store_key
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.codec import decode_fact, decode_operation, encode_fact, encode_operation
from linktools.ai.persistence.snapshots import _decode_snapshot_sequences
from linktools.ai.persistence.state import (
    STREAM_KEY_MAX_BYTES, OPERATION_KEY_MAX_BYTES, FactQuery, FactScanCursor,
    OperationQuery, StateTransaction, StoredFact, StoredOperation,
    capability_receipt_key, record_key, stream_key, validate_stream_key,
)

from .test_state_batch_delete_backends import _open_store, _record


def _tuple(*parts: str) -> bytes:
    return b"".join(len(value.encode()).to_bytes(2, "big") + value.encode() for value in parts)


def test_stream_components_preserve_exact_unicode_scope_and_empty_owner() -> None:
    namespace, tenant, owner = "界" * 42 + "ab", "界" * 42 + "ab", "界" * 42 + "ab"
    address = stream_key(namespace, tenant, "trace", "event", (owner,))
    assert address == _tuple("stream", namespace, tenant, "trace", "event", owner)
    assert stream_key("ns", "tenant", "conversation", "session_list_tenant", ()) == _tuple(
        "stream", "ns", "tenant", "conversation", "session_list_tenant",
    )
    assert stream_key("ns", "tenant", "trace", "event", ("a", "bc")) != stream_key(
        "ns", "tenant", "trace", "event", ("ab", "c"),
    )
    assert len({stream_key("ns", "tenant", "trace", "event", (value,))
                for value in ("A", "a", "é", "e\u0301")}) == 4
    for malformed in ("owner", ("owner", 1), (("owner",),)):
        with pytest.raises(TypeError):
            stream_key("ns", "tenant", "trace", "event", malformed)
    base = len(stream_key("ns", "tenant", "trace", "event", ("",)))
    assert len(stream_key("ns", "tenant", "trace", "event", ("x" * (STREAM_KEY_MAX_BYTES - base),))) == STREAM_KEY_MAX_BYTES
    with pytest.raises(ValueError):
        stream_key("ns", "tenant", "trace", "event", ("x" * (STREAM_KEY_MAX_BYTES + 1 - base),))


def test_stream_and_receipt_bounds_preserve_current_wire_and_private_repr() -> None:
    key = b"private-stream-identity".ljust(STREAM_KEY_MAX_BYTES, b"x")
    fact = StoredFact(key, 1, b"owner", "probe", None, None, {})
    operation = StoredOperation(b"private-operation".ljust(OPERATION_KEY_MAX_BYTES, b"x"), key, 1, "pending", False, {})
    assert decode_fact(encode_fact(fact)) == fact
    assert decode_operation(encode_operation(operation)) == operation
    assert _decode_snapshot_sequences([{"stream_key": key.hex(), "value": 3}]) == {key: 3}
    for value in (fact, operation, FactQuery(key), OperationQuery(stream_key=key), FactScanCursor(key, 1)):
        assert "private-stream-identity" not in repr(value)
    for bad in (b"", key + b"x", bytearray(key)):
        with pytest.raises(ValueError):
            validate_stream_key(bad)
    for decode, wire in ((decode_fact, encode_fact(fact)), (decode_operation, encode_operation(operation))):
        for bad in ("", (key + b"x").hex(), key.hex().upper()):
            with pytest.raises(AIError) as caught:
                decode(wire | {"stream": bad})
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError) as snapshot:
        _decode_snapshot_sequences([{"stream_key": (key + b"x").hex(), "value": 1}])
    assert snapshot.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    namespace = capability = write = "界" * 42 + "ab"
    tenant = run = "界" * 42 + "ab"
    receipt = capability_receipt_key(namespace, tenant, capability, (1 << 63) - 1, run, write)
    assert receipt == b"\x02" + _tuple("capability-write", namespace, tenant, capability, str((1 << 63) - 1), run, write)
    assert len(receipt) == 690 <= OPERATION_KEY_MAX_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sql"))
async def test_bare_store_stream_scope_family_counters_and_pagination(backend: str, tmp_path: Path) -> None:
    scopes = (("ns", "tenant", "trace"), ("other", "tenant", "trace"),
              ("ns", "other", "trace"), ("ns", "tenant", "execution"))
    families = ("execution", "task_event", "session_turn", "session_list_owner", "session_list_tenant",
                "operation", "history_transcript", "run_transcript", "transcript_pending_parts",
                "event", "checkpoint", "interaction")
    owners = {family: (() if family == "session_list_tenant" else
                       ("SESSION", "owner") if family == "operation" else
                       ("owner", "9223372036854775807") if family == "transcript_pending_parts" else
                       ("owner",)) for family in families}
    streams = [stream_key(*scope, family, owners[family]) for scope in scopes for family in families]
    # Primitive custom stores can also use the complete maximum address.
    streams += [b"x" * STREAM_KEY_MAX_BYTES, b"y" * STREAM_KEY_MAX_BYTES]
    assert len(set(streams)) == len(streams)
    owner = replace(_record(1), key=record_key("ns", "tenant", "trace", "probe", ("owner",)))
    facts = tuple(StoredFact(key, index, owner.key, "probe", "subject", None, {})
                  for key in streams for index in (1, 2, 3))
    operations = tuple(StoredOperation(b"op" + index.to_bytes(2, "big"), key, 1, "pending", False, {})
                       for index, key in enumerate(streams))
    async with _open_store(backend, tmp_path / backend) as store:
        async def seed(transaction: StateTransaction) -> None:
            await transaction.insert_record(owner)
            await transaction.insert_facts(facts)
            assert await transaction.reserve_sequences({key: 2 for key in streams}) == dict.fromkeys(streams, 2)
            for key in streams:
                assert await transaction.advance_sequence(key, 2) == 3
            for operation in operations:
                await transaction.insert_operation(operation)
        await store.mutate(seed)
        for key, operation in zip(streams, operations, strict=True):
            assert await store.read(lambda tx: tx.get_sequence(key)) == 3
            assert await store.read(lambda tx: tx.list_operations(OperationQuery(stream_key=key))) == (operation,)
            selected = await store.read(lambda tx: tx.list_facts(FactQuery(key, after_sequence=1, limit=1)))
            assert [(value.stream_key, value.sequence) for value in selected] == [(key, 2)]
            latest = await store.read(lambda tx: tx.list_facts(FactQuery(key, latest=True)))
            assert latest[0].sequence == 3
        found, cursor = [], None
        while page := await store.read(lambda tx: tx.scan_facts_page(after=cursor, limit=7)):
            found.extend(page)
            cursor = FactScanCursor(page[-1].stream_key, page[-1].sequence)
        assert found == sorted(facts, key=lambda value: (value.stream_key, value.sequence))
        found_counters, after = {}, None
        while page := await store.read(lambda tx: tx.scan_sequences_page(after=after, limit=7)):
            found_counters.update(page)
            after = max(page)
        assert found_counters == dict.fromkeys(streams, 3)
        await store.validate_integrity()
        async def reject_overflow(transaction: StateTransaction) -> None:
            with pytest.raises(ValueError):
                await transaction.next_sequence(b"x" * (STREAM_KEY_MAX_BYTES + 1))
        await store.mutate(reject_overflow)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ("ai_state_facts", "ai_state_operations", "ai_state_counters"))
async def test_sql_current_stream_readers_fail_closed_on_oversized_storage(table: str, tmp_path: Path) -> None:
    owner = _record(1)
    key = stream_key("ns", "tenant", "trace", "event", ("owner",))
    async with _open_store("sql", tmp_path) as store:
        async def seed(transaction: StateTransaction) -> None:
            await transaction.insert_record(owner)
            await transaction.insert_fact(StoredFact(key, 1, owner.key, "probe", None, None, {}))
            await transaction.insert_operation(StoredOperation(b"operation", key, 1, "pending", False, {}))
            await transaction.next_sequence(key)
        await store.mutate(seed)
        async with store.context.engine.begin() as connection:
            await connection.execute(text(f"UPDATE {table} SET stream_key = :key"), {"key": b"x" * (STREAM_KEY_MAX_BYTES + 1)})
        readers = {"ai_state_facts": lambda tx: tx.scan_facts(),
                   "ai_state_operations": lambda tx: tx.scan_operations(),
                   "ai_state_counters": lambda tx: tx.scan_sequences()}
        with pytest.raises(AIError) as caught:
            await store.read(readers[table])
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("changes", (
    {"key": b"wrong"}, {"stream_key": b"wrong"}, {"sequence": 1},
    {"state": "pending"}, {"compactable": True}, {"data": {"request_digest": "invalid"}},
))
def test_capability_receipt_reader_rejects_corrupt_identity_and_state(changes: dict[str, object]) -> None:
    from linktools.ai.persistence.capabilities import PreparedCapabilityWrites, _Write
    address = capability_receipt_key("ns", "tenant", "capability", 1, "run", "key")
    write = _Write("id", None, None, {}, address, "a" * 64, b"{}", False, True)
    receipt = StoredOperation(address, address, 0, "committed", False, {"request_digest": "a" * 64})
    PreparedCapabilityWrites._validate_receipt(receipt, write)
    with pytest.raises(AIError) as caught:
        PreparedCapabilityWrites._validate_receipt(replace(receipt, **changes), write)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert "capability" not in str(caught.value)
    with pytest.raises(AIError) as replay:
        PreparedCapabilityWrites._validate_receipt(replace(receipt, data={"request_digest": "b" * 64}), write)
    assert replay.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", ({"stream_key": b"wrong"}, {"sequence": 1}, {"compactable": True}))
async def test_partition_marker_reader_rejects_corrupt_physical_identity(changes: dict[str, object]) -> None:
    from linktools.ai.persistence._capability_marker import has_capability_partition_marker
    address = b"\x00capability-partitions"
    marker = StoredOperation(address, address, 0, "present", False, {"capability_partitions": True})

    class Reader:
        async def get_operation(self, key: bytes) -> StoredOperation:
            assert key == address
            return replace(marker, **changes)

    with pytest.raises(AIError) as caught:
        await has_capability_partition_marker(Reader())
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("index", (True, -1, 1 << 63))
def test_pending_transcript_index_owner_and_current_reader_enforce_signed_bound(index: int) -> None:
    from linktools.ai.persistence._message_root import NativeMessageRoot
    from linktools.ai.persistence._repository_common import _decode_enveloped_domain, domain_data
    from linktools.ai.persistence.contracts import TranscriptChunk, TranscriptOrigin
    chunk = TranscriptChunk("run", (1 << 63) - 1, 1, TranscriptOrigin.RAW, "codec", "a" * 64, 1,
                            NativeMessageRoot("a" * 32, native_store_key("b", "tenant"), ()))
    assert _decode_enveloped_domain(domain_data(chunk), TranscriptChunk) == chunk
    with pytest.raises(ValueError):
        replace(chunk, first_message_index=index)
    raw = domain_data(chunk)
    raw["value"]["payload"]["fields"]["first_message_index"] = index
    with pytest.raises(AIError) as caught:
        _decode_enveloped_domain(raw, TranscriptChunk)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
